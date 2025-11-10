import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np
from ModifiedCLIP import clip
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index
from datasets.vcoco_text_label import vcoco_hoi_text_label, vcoco_obj_text_label
from datasets.static_hico import HOI_IDX_TO_ACT_IDX

from ..backbone import build_backbone
from ..matcher import build_matcher
from .gen import build_gen


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


class HOICLIP(nn.Module):
    def __init__(self, backbone, transformer, num_queries, aux_loss=False, args=None):
        super().__init__()

        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed_h = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_o = nn.Embedding(num_queries, hidden_dim)
        self.pos_guided_embedd = nn.Embedding(num_queries, hidden_dim)
        self.hum_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.inter2verb = MLP(args.clip_embed_dim, args.clip_embed_dim // 2, args.clip_embed_dim, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_layers = self.args.dec_layers

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_model, self.preprocess = clip.load(self.args.clip_model)

        if self.args.dataset_file == 'hico':
            hoi_text_label = hico_text_label
            obj_text_label = hico_obj_text_label
            unseen_index = hico_unseen_index
        elif self.args.dataset_file == 'vcoco':
            hoi_text_label = vcoco_hoi_text_label
            obj_text_label = vcoco_obj_text_label
            unseen_index = None

        # 这里从 self.init_classifier_with_CLIP(...) 得到的返回值依次是：
        # clip_label:           所有 HOI（动作-物体对）在 CLIP 语义空间中的特征 embedding（tensor），用于 zero-shot 分类器权重
        # obj_clip_label:       所有 object 类别的 CLIP 特征 embedding（tensor），通常用于检测物体类别相关信息
        # v_linear_proj_weight: 通常是视觉分支输出到 CLIP embedding 空间的投影权重（tensor），比如线性层的权重
        # hoi_text:             HOI 类别对应的原始文本序列，长度为 HOI 类别数（如 600），元素为字符串
        # obj_text:             object 类别的原始文本列表，长度为物体类别数量+背景，元素为字符串
        # train_clip_label:     针对特定训练集 HOI 类别的 CLIP embedding（tensor），通常可能做了筛选、过滤或重排序
        clip_label, obj_clip_label, v_linear_proj_weight, hoi_text, obj_text, train_clip_label = \
            self.init_classifier_with_CLIP(hoi_text_label, obj_text_label, unseen_index, args.no_clip_cls_init)
        # num_obj_classes 表示物体类别总数（去除 "背景" 项或 "无" 类别），即 obj_text 列表长度减 1
        num_obj_classes = len(obj_text) - 1  # del nothing
        self.clip_visual_proj = v_linear_proj_weight

        self.hoi_class_fc = nn.Sequential(
            nn.Linear(hidden_dim, args.clip_embed_dim),
            nn.LayerNorm(args.clip_embed_dim),
        )

        if unseen_index:
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
        else:
            unseen_index_list = []

        if self.args.dataset_file == 'hico':
            verb2hoi_proj = torch.zeros(117, 600)  # HICO: 117 动词, 600 HOI
            # select_idx 为当前训练所有 HOI id（去除 zero-shot 的未见类别）
            select_idx = list(set([i for i in range(600)]) - set(unseen_index_list))
            # HOI_IDX_TO_ACT_IDX：将每个 HOI 映射到其对应的动词id
            for idx, v in enumerate(HOI_IDX_TO_ACT_IDX):
                verb2hoi_proj[v][idx] = 1.0  # 第idx个HOI属于第v个动词
            # self.verb2hoi_proj 只含训练用的 HOI（剔除unseen）
            self.verb2hoi_proj = nn.Parameter(verb2hoi_proj[:, select_idx], requires_grad=False)
            # self.verb2hoi_proj_eval 保留所有 HOI，用于评估时未删减的映射
            self.verb2hoi_proj_eval = nn.Parameter(verb2hoi_proj, requires_grad=False)

            # 动词预测线性变换层，输出维度为动词类别数，权重由文件加载
            self.verb_projection = nn.Linear(args.clip_embed_dim, 117, bias=False)
            self.verb_projection.weight.data = torch.load(args.verb_pth, map_location='cpu')
            self.verb_weight = args.verb_weight
        else:
            verb2hoi_proj = torch.zeros(29, 263)
            for i in vcoco_hoi_text_label.keys():
                verb2hoi_proj[i[0]][i[1]] = 1

            self.verb2hoi_proj = nn.Parameter(verb2hoi_proj, requires_grad=False)
            self.verb_projection = nn.Linear(args.clip_embed_dim, 29, bias=False)
            self.verb_projection.weight.data = torch.load(args.verb_pth, map_location='cpu')
            self.verb_weight = args.verb_weight

        if args.with_clip_label:
            if args.fix_clip_label:
                self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text), bias=False)
                self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)
                for i in self.visual_projection.parameters():
                    i.require_grads = False
            else:
                self.visual_projection = nn.Linear(args.clip_embed_dim, len(hoi_text))
                self.visual_projection.weight.data = train_clip_label / train_clip_label.norm(dim=-1, keepdim=True)

            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default':
                self.eval_visual_projection = nn.Linear(args.clip_embed_dim, 600, bias=False)
                self.eval_visual_projection.weight.data = clip_label / clip_label.norm(dim=-1, keepdim=True)
        else:
            self.hoi_class_embedding = nn.Linear(args.clip_embed_dim, len(hoi_text))

        if args.with_obj_clip_label:
            self.obj_class_fc = nn.Sequential(
                nn.Linear(hidden_dim, args.clip_embed_dim),
                nn.LayerNorm(args.clip_embed_dim),
            )
            if args.fix_clip_label:
                self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1, bias=False)
                self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
                for i in self.obj_visual_projection.parameters():
                    i.require_grads = False
            else:
                self.obj_visual_projection = nn.Linear(args.clip_embed_dim, num_obj_classes + 1)
                self.obj_visual_projection.weight.data = obj_clip_label / obj_clip_label.norm(dim=-1, keepdim=True)
        else:
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

        self.transformer.hoi_cls = clip_label / clip_label.norm(dim=-1, keepdim=True)

        # VPT和场景提示词相关参数
        self.VPT_length = getattr(args, 'VPT_length', 8)
        self.img_scene_num = getattr(args, 'img_scene_num', 4)
        self.VPT_low_rank = getattr(args, 'VPT_low_rank', False)
        self.low_rank = getattr(args, 'low_rank', True)
        self.pattern_num = getattr(args, 'pattern_num', 2)
        vision_width = self.clip_model.visual.conv1.weight.shape[0]
        
        if self.VPT_length > 0:
            if self.VPT_low_rank:
                self.VPT_u = nn.Parameter(torch.empty(1, self.VPT_length))
                self.VPT_v = nn.Parameter(torch.empty(1, vision_width))
            else:
                self.VPT = nn.Parameter(torch.empty(self.VPT_length, vision_width))
        
        if self.img_scene_num > 0:
            if self.low_rank:
                self.img_scene_prompt_u = nn.Parameter(torch.empty(self.img_scene_num, 1, self.VPT_length))
                self.img_scene_prompt_v = nn.Parameter(torch.empty(self.img_scene_num, 1, vision_width))
            else:
                self.img_scene_prompt = nn.Parameter(torch.empty(self.img_scene_num, self.VPT_length, vision_width))
            
            self.img_scene_prompt_to_key = nn.Sequential(
                nn.Linear(self.VPT_length, self.VPT_length // 2),
                nn.GELU(),
                nn.Linear(self.VPT_length // 2, 1) 
            )
            self.img_scene_prompt_to_key2 = nn.Sequential(
                nn.Linear(vision_width, vision_width // 2),
                nn.GELU(),
                nn.Linear(vision_width // 2, args.clip_embed_dim)
            )
        
        # 用于从backbone特征提取fingerprint的投影层
        self.img_fingerprint_proj = nn.Linear(hidden_dim, args.clip_embed_dim)
        
        self.hidden_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embedd.weight)
        
        # 初始化VPT和场景提示词参数
        if self.VPT_length > 0:
            if self.VPT_low_rank:
                nn.init.normal_(self.VPT_u, std=0.01)
                nn.init.normal_(self.VPT_v, std=0.01)
            else:
                nn.init.normal_(self.VPT, std=0.01)
        
        if self.img_scene_num > 0:
            if self.low_rank:
                nn.init.normal_(self.img_scene_prompt_u, std=0.01)
                nn.init.normal_(self.img_scene_prompt_v, std=0.01)
            else:
                nn.init.normal_(self.img_scene_prompt, std=0.01)
            nn.init.normal_(self.img_scene_prompt_to_key[0].weight, std=0.01)
            nn.init.normal_(self.img_scene_prompt_to_key[2].weight, std=0.01)
            nn.init.normal_(self.img_scene_prompt_to_key2[0].weight, std=0.01)
            nn.init.normal_(self.img_scene_prompt_to_key2[2].weight, std=0.01)
        
        # 初始化fingerprint投影层
        nn.init.normal_(self.img_fingerprint_proj.weight, std=0.01)
        nn.init.zeros_(self.img_fingerprint_proj.bias)

    def init_classifier_with_CLIP(self, hoi_text_label, obj_text_label, unseen_index, no_clip_cls_init=False):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_inputs = torch.cat([clip.tokenize(hoi_text_label[id]) for id in hoi_text_label.keys()])
        if self.args.del_unseen and unseen_index is not None:
            hoi_text_label_del = {}
            unseen_index_list = unseen_index.get(self.args.zero_shot_type, [])
            for idx, k in enumerate(hoi_text_label.keys()):
                if idx in unseen_index_list:
                    continue
                else:
                    hoi_text_label_del[k] = hoi_text_label[k]
        else:
            hoi_text_label_del = hoi_text_label.copy()
        text_inputs_del = torch.cat(
            [clip.tokenize(hoi_text_label[id]) for id in hoi_text_label_del.keys()])

        obj_text_inputs = torch.cat([clip.tokenize(obj_text[1]) for obj_text in obj_text_label])
        clip_model = self.clip_model
        clip_model.to(device)
        with torch.no_grad():
            text_embedding = clip_model.encode_text(text_inputs.to(device))
            text_embedding_del = clip_model.encode_text(text_inputs_del.to(device))
            obj_text_embedding = clip_model.encode_text(obj_text_inputs.to(device))
            v_linear_proj_weight = clip_model.visual.proj.detach()

        if not no_clip_cls_init:
            print('\nuse clip text encoder to init classifier weight\n')
            return text_embedding.float(), obj_text_embedding.float(), v_linear_proj_weight.float(), \
                   hoi_text_label_del, obj_text_inputs, text_embedding_del.float()
        else:
            print('\nnot use clip text encoder to init classifier weight\n')
            return torch.randn_like(text_embedding.float()), torch.randn_like(
                obj_text_embedding.float()), torch.randn_like(v_linear_proj_weight.float()), \
                   hoi_text_label_del, obj_text_inputs, torch.randn_like(text_embedding_del.float())

    def forward(self, samples: NestedTensor, is_training=True, clip_input=None, targets=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        
        # 构造场景提示词
        img_scene_prompts = None
        bs = src.shape[0]
        device = src.device
        vision_width = self.clip_model.visual.conv1.weight.shape[0]
        
        # 从backbone特征提取图像fingerprint
        # 使用全局平均池化
        backbone_feat = F.adaptive_avg_pool2d(src, (1, 1)).flatten(1)  # [B, C]
        cur_img_fingerprints = self.img_fingerprint_proj(backbone_feat)  # [B, clip_embed_dim]
        
        if self.VPT_length > 0:
            if self.VPT_low_rank:
                VPT = self.VPT_u.transpose(0, 1).contiguous() @ self.VPT_v
            else:
                VPT = self.VPT
            
            if self.VPT_length > 0 and self.img_scene_num == 0:
                # 使用VPT作为场景提示词
                img_scene_prompts = VPT.unsqueeze(0).repeat(bs, 1, 1).to(device)
            
            if self.img_scene_num > 0:
                if self.low_rank:
                    img_scene_prompts = self.img_scene_prompt_u.transpose(1, 2).contiguous() @ self.img_scene_prompt_v
                else:
                    img_scene_prompts = self.img_scene_prompt
                # img_scene_prompts: [img_scene_num, VPT_length, vision_width]
                # 确保VPT和img_scene_prompts在正确的设备上
                VPT = VPT.to(device)
                img_scene_prompts = img_scene_prompts.to(device)
                img_scene_prompts = img_scene_prompts * VPT.unsqueeze(0)  # [img_scene_num, VPT_length, vision_width]
                # 将img_scene_prompts投影到key空间用于注意力计算
                # 参考代码：img_scene_prompt_key = self.img_scene_prompt_to_key(self.img_scene_prompt_to_key2(img_scene_prompts).transpose(1, 2).contiguous()).squeeze()
                # img_scene_prompts: [img_scene_num, VPT_length, vision_width]
                # img_scene_prompt_to_key2: 对vision_width维度操作，输出embed_dim
                # 输入 [img_scene_num, VPT_length, vision_width] -> 输出 [img_scene_num, VPT_length, embed_dim]
                img_scene_prompts_proj = self.img_scene_prompt_to_key2(img_scene_prompts)  # [img_scene_num, VPT_length, embed_dim]
                # transpose: [img_scene_num, VPT_length, embed_dim] -> [img_scene_num, embed_dim, VPT_length]
                img_scene_prompts_proj_t = img_scene_prompts_proj.transpose(1, 2).contiguous()  # [img_scene_num, embed_dim, VPT_length]
                # img_scene_prompt_to_key: 对VPT_length维度操作，输出1
                # 输入 [img_scene_num, embed_dim, VPT_length] -> 输出 [img_scene_num, embed_dim, 1]
                img_scene_prompt_key = self.img_scene_prompt_to_key(img_scene_prompts_proj_t).squeeze(-1)  # [img_scene_num, embed_dim]
                # 使用cur_img_fingerprints作为query，img_scene_prompt_key作为key，选择top-k场景提示词
                attn_scores = F.softmax(cur_img_fingerprints.float() @ img_scene_prompt_key.T, dim=-1)  # [B, img_scene_num]
                top_scores, top_indices = attn_scores.topk(self.pattern_num, dim=-1)  # [B, pattern_num]
                # img_scene_prompts[top_indices]: [B, pattern_num, VPT_length, vision_width]
                # top_scores: [B, pattern_num] -> [B, pattern_num, 1, 1]
                img_scene_prompts = (top_scores.unsqueeze(-1).unsqueeze(-1) * img_scene_prompts[top_indices]).sum(dim=1)  # [B, VPT_length, vision_width]
        
        # 这里的输出的形状如下：
        # h_hs: [num_decoder_layers, batch_size, num_queries, hidden_dim]，对应每层 decoder 的 human 查询向量
        # o_hs: [num_decoder_layers, batch_size, num_queries, hidden_dim]，对应 object 查询向量
        # inter_hs: [num_decoder_layers, batch_size, num_queries, hidden_dim]，对应交互 (interaction) 查询向量
        # clip_cls_feature: [batch_size, num_queries, clip_feature_dim]，用于与文本特征做匹配
        # clip_hoi_score: [batch_size, num_queries, num_hoi_classes]，CLIP 分数 (zero-shot 分类分数)
        # clip_visual: [batch_size, num_queries, clip_feature_dim]，CLIP 的 visual 特征（通常归一化后的）
        h_hs, o_hs, inter_hs, clip_cls_feature, clip_hoi_score, clip_visual = self.transformer(
            self.input_proj(src), mask,
            self.query_embed_h.weight,
            self.query_embed_o.weight,
            self.pos_guided_embedd.weight,
            pos[-1], self.clip_model, self.clip_visual_proj, clip_input, img_scene_prompts
        )

        outputs_sub_coord = self.hum_bbox_embed(h_hs).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(o_hs).sigmoid()

        if self.args.with_obj_clip_label:
            obj_logit_scale = self.obj_logit_scale.exp()
            o_hs = self.obj_class_fc(o_hs)
            o_hs = o_hs / o_hs.norm(dim=-1, keepdim=True)
            outputs_obj_class = obj_logit_scale * self.obj_visual_projection(o_hs)
        else:
            outputs_obj_class = self.obj_class_embed(o_hs)

        # 这一块代码用于根据模型设置不同的分类方式生成 HOI (Human-Object Interaction) 分类输出。
        # 如果开启了 with_clip_label，说明要用 CLIP 风格的 zero-shot 分类器权重，且融合动词头结果:
        if self.args.with_clip_label:
            logit_scale = self.logit_scale.exp()
            # outputs_inter_hs 保留归一化前的交互特征，用于后续 mimicking 辅助损失或其他用途。
            outputs_inter_hs = inter_hs.clone()
            # inter2verb 将交互特征投影到动词特征空间，用于 verb 分支分类。
            verb_hs = self.inter2verb(inter_hs)
            # 对交互特征/动词特征做归一化，保证特征空间内点积是余弦相似度，有利于与文本特征对齐。
            inter_hs = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
            verb_hs = verb_hs / verb_hs.norm(dim=-1, keepdim=True)
            # 对 hico 数据集做特殊处理：zero_shot_type 非默认，且在评估或测试阶段时，用 eval_visual_projection 和 verb2hoi_proj_eval；
            # 否则，使用标准 visual_projection 和 verb2hoi_proj。
            if self.args.dataset_file == 'hico' and self.args.zero_shot_type != 'default' \
                    and (self.args.eval or not is_training):
                outputs_hoi_class = logit_scale * self.eval_visual_projection(inter_hs)
                outputs_verb_class = logit_scale * self.verb_projection(verb_hs) @ self.verb2hoi_proj_eval
                outputs_hoi_class = outputs_hoi_class + outputs_verb_class * self.verb_weight
            else:
                outputs_hoi_class = logit_scale * self.visual_projection(inter_hs)
                outputs_verb_class = logit_scale * self.verb_projection(verb_hs) @ self.verb2hoi_proj
                outputs_hoi_class = outputs_hoi_class + outputs_verb_class * self.verb_weight
        else:
            inter_hs = self.hoi_class_fc(inter_hs)
            outputs_inter_hs = inter_hs.clone()
            outputs_hoi_class = self.hoi_class_embedding(inter_hs)

        out = {'pred_hoi_logits': outputs_hoi_class[-1], 'pred_obj_logits': outputs_obj_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1], 'clip_visual': clip_visual,
               'clip_cls_feature': clip_cls_feature, 'hoi_feature': inter_hs[-1], 'clip_logits': clip_hoi_score}

        # 这是做什么的？
        # 如果开启了 with_mimic（即开启“mimic”辅助损失），则将 outputs_inter_hs[-1]（交互特征最后一层）保存到输出字典 out['inter_memory']，供后面 loss 计算用。
        if self.args.with_mimic:
            out['inter_memory'] = outputs_inter_hs[-1]
        # 如果开启了 auxiliary loss（辅助损失，多层监督），则需要为所有 decoder 层收集辅助输出。
        if self.aux_loss:
            # 如果 with_mimic 也开启了，则 aux_mimic 就是所有 decoder 层的中间交互特征 outputs_inter_hs（shape: [decoder层数, ...]）
            if self.args.with_mimic:
                aux_mimic = outputs_inter_hs
            # 否则不需要 mimic 辅助监督，aux_mimic 置为 None
            else:
                aux_mimic = None

            # 通过 _set_aux_loss_triplet 函数，把 HOI logits、object logits、两个 bbox 预测和 aux_mimic 打包给 out['aux_outputs']，后续多层 loss 使用
            out['aux_outputs'] = self._set_aux_loss_triplet(outputs_hoi_class, outputs_obj_class,
                                                            outputs_sub_coord, outputs_obj_coord,
                                                            aux_mimic)

        return out

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_hoi_class, outputs_obj_class,
                              outputs_sub_coord, outputs_obj_coord, outputs_inter_hs=None):

        if outputs_hoi_class.shape[0] == 1:
            outputs_hoi_class = outputs_hoi_class.repeat(self.dec_layers, 1, 1, 1)
        aux_outputs = {'pred_hoi_logits': outputs_hoi_class[-self.dec_layers: -1],
                       'pred_obj_logits': outputs_obj_class[-self.dec_layers: -1],
                       'pred_sub_boxes': outputs_sub_coord[-self.dec_layers: -1],
                       'pred_obj_boxes': outputs_obj_coord[-self.dec_layers: -1],
                       }
        if outputs_inter_hs is not None:
            aux_outputs['inter_memory'] = outputs_inter_hs[-self.dec_layers: -1]
        outputs_auxes = []
        for i in range(self.dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.with_mimic:
            self.clip_model, _ = clip.load(args.clip_model, device=device)
        else:
            self.clip_model = None
        self.alpha = args.alpha

    # 计算物体（object）分类损失的函数
    # 该函数实现目标检测/HOI中的物体识别损失与准确率统计
    # 参数说明：
    # outputs:   网络前向输出的字典，需包含 'pred_obj_logits'，即所有预测queries的物体分类logits（shape: [batch_size, num_queries, num_obj_classes+1]）
    # targets:   标注信息，每个元素是一个dict，需包含'obj_labels'（实际物体类别标签，int索引，shape: [num_obj_instances_per_img]）
    # indices:   matcher匹配好的索引对应，每对为(预测idx, 标注idx)，与targets一一对应
    # num_interactions: 当前用于归一化损失的交互总数（一般取未归一化，用于后续sum／mean归一化）
    # log:       是否返回准确率指标
    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        # 1. 检查输出dict中有物体分类logits
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        # 2. 获取被matcher分配上的query索引对
        idx = self._get_src_permutation_idx(indices)

        # 3. 将所有匹配到的GT物体标签（索引）拼接到一起，组成target_classes_o（一维向量）
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])

        # 4. 生成和所有预测logits shape一致的目标类别tensor，初始填充为背景（num_obj_classes），在匹配位置填充上GT索引
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # 5. 计算交叉熵损失（已包含背景类别），采用空类别（背景）权重调整；注意PyTorch交叉熵期望shape：[batch, num_classes, num_queries]
        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        # 6. 可选，记录分类准确率，默认log=True
        if log:
            # 计算被matcher分配到的query中的物体分类准确率
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # 该函数用于衡量模型在每张图片中预测的“物体实例数”和标注的真实“物体实例数”之间的差异，属于计数准确性指标（cardinality error）。
    # 主要流程为：
    #   1. 对 outputs['pred_obj_logits'] 取 argmax 得到每个 query 预测的类别索引，统计非背景（即 num_obj_classes 以外）的个数，视为该图片预测出的物体数量（card_pred）。
    #   2. 从 targets 提取每张图片真实的物体标签数量（tgt_lengths）。
    #   3. 用 L1 损失计算模型预测物体数与真实数量的平均绝对误差（card_err）。
    #   4. 返回 {'obj_cardinality_error': card_err}。
    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        # 统计每张图片被预测为非背景（即真正预测为某一物体类别）的 query 数量
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # L1 损失作为“物体实例数”误差
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses
    # 该函数用于衡量模型在每张图片中预测的“动词实例数”和标注的真实“动词实例数”之间的差异，属于计数准确性指标（cardinality error）。
    # 主要流程为：
    #   1. 对 outputs['pred_verb_logits'] 取 argmax 得到每个 query 预测的类别索引，统计非背景（即 num_verb_classes 以外）的个数，视为该图片预测出的动词数量（card_pred）。
    #   2. 从 targets 提取每张图片真实的动词标签数量（tgt_lengths）。
    #   3. 用 L1 损失计算模型预测动词数与真实数量的平均绝对误差（card_err）。
    #   4. 返回 {'verb_cardinality_error': card_err}。
    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        # 1. 检查输出dict中有动词分类logits
        assert 'pred_verb_logits' in outputs
        # 2. 获取被matcher分配上的query索引对
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        # 3. 将所有匹配到的GT动词标签（索引）拼接到一起，组成target_classes_o（一维向量）
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        # 4. 生成和所有预测logits shape一致的目标类别tensor，初始填充为背景（num_verb_classes），在匹配位置填充上GT索引
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        # 5. 计算交叉熵损失（已包含背景类别），采用空类别（背景）权重调整；注意PyTorch交叉熵期望shape：[batch, num_classes, num_queries]
        src_logits = src_logits.sigmoid()
        # 6. 计算动词分类损失
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_hoi_labels(self, outputs, targets, indices, num_interactions, topk=5):
        assert 'pred_hoi_logits' in outputs
        src_logits = outputs['pred_hoi_logits']
        dtype = src_logits.dtype

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['hoi_labels'][J] for t, (_, J) in zip(targets, indices)]).to(dtype)
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        src_logits = _sigmoid(src_logits)
        loss_hoi_ce = self._neg_loss(src_logits, target_classes, weights=None, alpha=self.alpha)
        losses = {'loss_hoi_labels': loss_hoi_ce}

        _, pred = src_logits[idx].topk(topk, 1, True, True)
        acc = 0.0
        for tid, target in enumerate(target_classes_o):
            tgt_idx = torch.where(target == 1)[0]
            if len(tgt_idx) == 0:
                continue
            acc_pred = 0.0
            for tgt_rel in tgt_idx:
                acc_pred += (tgt_rel in pred[tid])
            acc += acc_pred / len(tgt_idx)
        rel_labels_error = 100 - 100 * acc / max(len(target_classes_o), 1)
        losses['hoi_class_error'] = torch.from_numpy(np.array(
            rel_labels_error)).to(src_logits.device).float()
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                    exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def mimic_loss(self, outputs, targets, indices, num_interactions):
        src_feats = outputs['inter_memory']
        src_feats = torch.mean(src_feats, dim=1)

        target_clip_inputs = torch.cat([t['clip_inputs'].unsqueeze(0) for t in targets])
        with torch.no_grad():
            target_clip_feats = self.clip_model.encode_image(target_clip_inputs)
        loss_feat_mimic = F.l1_loss(src_feats, target_clip_feats)
        losses = {'loss_feat_mimic': loss_feat_mimic}
        return losses
    def reconstruction_loss(self, outputs, targets, indices, num_interactions):
        raw_feature = outputs['clip_cls_feature']
        hoi_feature = outputs['hoi_feature']

        loss_rec = F.l1_loss(raw_feature, hoi_feature)
        return {'loss_rec': loss_rec}

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        if 'pred_hoi_logits' in outputs.keys():
            loss_map = {
                'hoi_labels': self.loss_hoi_labels,
                'obj_labels': self.loss_obj_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
                'feats_mimic': self.mimic_loss,
                'rec_loss': self.reconstruction_loss
            }
        else:
            loss_map = {
                'obj_labels': self.loss_obj_labels,
                'obj_cardinality': self.loss_obj_cardinality,
                'verb_labels': self.loss_verb_labels,
                'sub_obj_boxes': self.loss_sub_obj_boxes,
            }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)
    # 该函数是模型前向传播的核心，负责计算和返回所有损失。
    # 主要流程为：
    #   1. 分离出不含辅助输出的主输出字典（outputs_without_aux），辅助输出通常在 'aux_outputs' 中。
    #   2. 使用 matcher 计算主输出与目标之间的匹配索引对（indices）。
    #   3. 统计当前 batch 中交互总数（num_interactions），用于后续损失归一化。
    #   4. 遍历所有损失类型，调用 get_loss 计算每种损失的值，并累加到 losses 字典中。
    #   5. 如果存在辅助输出（即多层解码器），则遍历每一层辅助输出，重复上述过程，损失值加后缀从0到(dec_layers-2)。
    #   6. 返回包含所有损失的字典 losses。
    def forward(self, outputs, targets):
        # 1. 分离出不含辅助输出的主输出字典（outputs_without_aux），辅助输出通常在 'aux_outputs' 中。
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['hoi_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss =='rec_loss':
                        continue
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    # 该函数是模型后处理的核心，负责将模型预测结果转换为可读的格式。
    # 主要流程为：
    #   1. 从 outputs 字典中提取预测的 HOI 得分、物体得分、物体标签、子框和物体框。
    #   2. 根据目标图像尺寸（target_sizes），将预测框从归一化坐标转换为像素坐标。
    #   3. 创建结果列表，每个元素包含预测的标签、框和得分。
    #   4. 为每个结果添加 HOI 得分、物体得分、CLIP 视觉特征、子对象ID、物体ID和CLIP 分类得分。
    #   5. 返回包含所有结果的字典 results。
class PostProcessHOITriplet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_hoi_logits = outputs['pred_hoi_logits']
        out_obj_logits = outputs['pred_obj_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']
        clip_visual = outputs['clip_visual']
        clip_logits = outputs['clip_logits']

        assert len(out_hoi_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.sigmoid()
        obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)[1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(hoi_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(hoi_scores)):
            hs, os, ol, sb, ob = hoi_scores[index], obj_scores[index], obj_labels[index], sub_boxes[index], obj_boxes[
                index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            ids = torch.arange(b.shape[0])

            results[-1].update({'hoi_scores': hs.to('cpu'), 'obj_scores': os.to('cpu'), 'clip_visual': clip_visual[index].to('cpu'),
                                'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:], 'clip_logits': clip_logits[index].to('cpu')})

        return results


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    gen = build_gen(args)

    model = HOICLIP(
        backbone,
        gen,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    # 解释各个变量
    weight_dict = {}  # 损失项的权重字典，用于在总损失中加权不同的损失
    if args.with_clip_label:
        # loss_hoi_labels: HOI类别（verb-object对）多标签损失的权重
        # loss_obj_ce: 物体检测的交叉熵损失的权重
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef
    else:
        # 与上面类似，不同分支做保留（通常为兼容后续扩展）
        weight_dict['loss_hoi_labels'] = args.hoi_loss_coef
        weight_dict['loss_obj_ce'] = args.obj_loss_coef

    # loss_sub_bbox: 主体（human）边框回归损失的权重
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    # loss_obj_bbox: 客体（object）边框回归损失的权重
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    # loss_sub_giou: 主体（human）边框GIoU损失的权重
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    # loss_obj_giou: 客体（object）边框GIoU损失的权重
    weight_dict['loss_obj_giou'] = args.giou_loss_coef

    if args.with_mimic:
        # loss_feat_mimic: 特征对齐/模仿损失(如和CLIP对齐)的权重
        weight_dict['loss_feat_mimic'] = args.mimic_loss_coef

    if args.with_rec_loss:
        # loss_rec: 重建损失（如自监督或特殊任务）的权重
        weight_dict['loss_rec'] = args.rec_loss_coef

    if args.aux_loss:
        # 如果使用辅助损失（多层解码器的每层均有损失），每个解码层分别加后缀从0到(dec_layers-2)
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # losses: 用于SetCriterion进行前向损失计算的损失种类列表
    losses = ['hoi_labels', 'obj_labels', 'sub_obj_boxes']  # 'hoi_labels'为HOI多标签损失, 'obj_labels'为物体分类损失, 'sub_obj_boxes'为人-物框损失
    if args.with_mimic:
        # feats_mimic: 特征模仿损失
        losses.append('feats_mimic')

    if args.with_rec_loss:
        # rec_loss: 重建损失
        losses.append('rec_loss')

    # 创建损失计算器 SetCriterionHOI，参数包括：
    # num_obj_classes: 物体类别总数
    # num_queries: 查询数量（如目标检测框数）
    # num_verb_classes: 动词类别数
    # matcher: 匹配器，用于将预测和真实框进行匹配
    # weight_dict: 损失权重字典
    # eos_coef: 结束符号（EOS）的权重系数
    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOITriplet(args)}

    return model, criterion, postprocessors
