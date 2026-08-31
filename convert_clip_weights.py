"""
把 HuggingFace 的 CLIP ViT-L/14 权重转换为 OpenAI 原版格式。

HF 格式: vision_model.*, text_model.*, visual_projection, text_projection
OpenAI:  visual.*, transformer.*, visual.proj, text_projection, positional_embedding, ...

用法:
    python3 convert_clip_weights.py <hf_weight.pt> <output.pt>
"""
import torch
import sys
import os


def convert_hf_to_openai(hf_state: dict) -> dict:
    """把 HF CLIPModel state_dict 转成 OpenAI clip 格式"""
    openai_state = {}

    # 如果是嵌套的, 取出
    if 'state_dict' in hf_state:
        hf_state = hf_state['state_dict']
    elif 'model' in hf_state and isinstance(hf_state['model'], dict):
        hf_state = hf_state['model']

    # ===== 1. logit_scale =====
    if 'logit_scale' in hf_state:
        openai_state['logit_scale'] = hf_state['logit_scale']

    # ===== 2. Visual encoder (vision_model.* -> visual.*) =====
    # ViT-L/14 是 ViT 结构 (不是 ResNet), 所以走 vit 分支

    # visual.conv1: HF 的 patch_embedding
    if 'vision_model.embeddings.patch_embedding.weight' in hf_state:
        openai_state['visual.conv1.weight'] = hf_state['vision_model.embeddings.patch_embedding.weight']

    # visual.positional_embedding: HF 的 position_embedding (注意: HF 包含 CLS token)
    if 'vision_model.embeddings.position_embedding.weight' in hf_state:
        openai_state['visual.positional_embedding'] = hf_state['vision_model.embeddings.position_embedding.weight']

    # visual.ln_pre: HF 的 pre_layrnorm
    if 'vision_model.pre_layrnorm.weight' in hf_state:
        openai_state['visual.ln_pre.weight'] = hf_state['vision_model.pre_layrnorm.weight']
    if 'vision_model.pre_layrnorm.bias' in hf_state:
        openai_state['visual.ln_pre.bias'] = hf_state['vision_model.pre_layrnorm.bias']

    # visual.transformer.resblocks.{i}.*
    # HF: vision_model.encoder.layers.{i}.*
    # OpenAI: visual.transformer.resblocks.{i}.*
    num_visual_layers = 0
    for key in hf_state.keys():
        if key.startswith('vision_model.encoder.layers.'):
            layer_idx = key.split('.')[3]
            num_visual_layers = max(num_visual_layers, int(layer_idx) + 1)

    for i in range(num_visual_layers):
        hf_prefix = f'vision_model.encoder.layers.{i}'
        oa_prefix = f'visual.transformer.resblocks.{i}'

        # attention: HF 拆成 q/k/v/out_proj, OpenAI 合并成 in_proj
        # q_proj
        if f'{hf_prefix}.self_attn.q_proj.weight' in hf_state:
            q_w = hf_state[f'{hf_prefix}.self_attn.q_proj.weight']
            k_w = hf_state[f'{hf_prefix}.self_attn.k_proj.weight']
            v_w = hf_state[f'{hf_prefix}.self_attn.v_proj.weight']
            openai_state[f'{oa_prefix}.attn.in_proj_weight'] = torch.cat([q_w, k_w, v_w], dim=0)

            q_b = hf_state[f'{hf_prefix}.self_attn.q_proj.bias']
            k_b = hf_state[f'{hf_prefix}.self_attn.k_proj.bias']
            v_b = hf_state[f'{hf_prefix}.self_attn.v_proj.bias']
            openai_state[f'{oa_prefix}.attn.in_proj_bias'] = torch.cat([q_b, k_b, v_b], dim=0)

        if f'{hf_prefix}.self_attn.out_proj.weight' in hf_state:
            openai_state[f'{oa_prefix}.attn.out_proj.weight'] = hf_state[f'{hf_prefix}.self_attn.out_proj.weight']
        if f'{hf_prefix}.self_attn.out_proj.bias' in hf_state:
            openai_state[f'{oa_prefix}.attn.out_proj.bias'] = hf_state[f'{hf_prefix}.self_attn.out_proj.bias']

        # layer_norm1 -> ln_1
        if f'{hf_prefix}.layer_norm1.weight' in hf_state:
            openai_state[f'{oa_prefix}.ln_1.weight'] = hf_state[f'{hf_prefix}.layer_norm1.weight']
        if f'{hf_prefix}.layer_norm1.bias' in hf_state:
            openai_state[f'{oa_prefix}.ln_1.bias'] = hf_state[f'{hf_prefix}.layer_norm1.bias']

        # mlp fc1 -> c_fc
        if f'{hf_prefix}.mlp.fc1.weight' in hf_state:
            openai_state[f'{oa_prefix}.mlp.c_fc.weight'] = hf_state[f'{hf_prefix}.mlp.fc1.weight']
        if f'{hf_prefix}.mlp.fc1.bias' in hf_state:
            openai_state[f'{oa_prefix}.mlp.c_fc.bias'] = hf_state[f'{hf_prefix}.mlp.fc1.bias']

        # mlp fc2 -> c_proj
        if f'{hf_prefix}.mlp.fc2.weight' in hf_state:
            openai_state[f'{oa_prefix}.mlp.c_proj.weight'] = hf_state[f'{hf_prefix}.mlp.fc2.weight']
        if f'{hf_prefix}.mlp.fc2.bias' in hf_state:
            openai_state[f'{oa_prefix}.mlp.c_proj.bias'] = hf_state[f'{hf_prefix}.mlp.fc2.bias']

        # layer_norm2 -> ln_2
        if f'{hf_prefix}.layer_norm2.weight' in hf_state:
            openai_state[f'{oa_prefix}.ln_2.weight'] = hf_state[f'{hf_prefix}.layer_norm2.weight']
        if f'{hf_prefix}.layer_norm2.bias' in hf_state:
            openai_state[f'{oa_prefix}.ln_2.bias'] = hf_state[f'{hf_prefix}.layer_norm2.bias']

    # visual.class_embedding: HF 的 class_embedding
    if 'vision_model.embeddings.class_embedding' in hf_state:
        openai_state['visual.class_embedding'] = hf_state['vision_model.embeddings.class_embedding']

    # visual.ln_post: HF 的 post_layernorm
    if 'vision_model.post_layernorm.weight' in hf_state:
        openai_state['visual.ln_post.weight'] = hf_state['vision_model.post_layernorm.weight']
    if 'vision_model.post_layernorm.bias' in hf_state:
        openai_state['visual.ln_post.bias'] = hf_state['vision_model.post_layernorm.bias']

    # visual.proj: OpenAI 的 visual.proj 是 [width, output_dim] = [1024, 768]
    # HF 的 visual_projection.weight 是 [output_dim, width] = [768, 1024], 需要转置
    if 'visual_projection.weight' in hf_state:
        openai_state['visual.proj'] = hf_state['visual_projection.weight'].T

    # ===== 3. Text encoder (text_model.* -> transformer.*) =====
    # token_embedding
    if 'text_model.embeddings.token_embedding.weight' in hf_state:
        openai_state['token_embedding.weight'] = hf_state['text_model.embeddings.token_embedding.weight']

    # positional_embedding
    if 'text_model.embeddings.position_embedding.weight' in hf_state:
        openai_state['positional_embedding'] = hf_state['text_model.embeddings.position_embedding.weight']

    # transformer.resblocks.{i}.*
    num_text_layers = 0
    for key in hf_state.keys():
        if key.startswith('text_model.encoder.layers.'):
            layer_idx = key.split('.')[3]
            num_text_layers = max(num_text_layers, int(layer_idx) + 1)

    for i in range(num_text_layers):
        hf_prefix = f'text_model.encoder.layers.{i}'
        oa_prefix = f'transformer.resblocks.{i}'

        # attention
        if f'{hf_prefix}.self_attn.q_proj.weight' in hf_state:
            q_w = hf_state[f'{hf_prefix}.self_attn.q_proj.weight']
            k_w = hf_state[f'{hf_prefix}.self_attn.k_proj.weight']
            v_w = hf_state[f'{hf_prefix}.self_attn.v_proj.weight']
            openai_state[f'{oa_prefix}.attn.in_proj_weight'] = torch.cat([q_w, k_w, v_w], dim=0)

            q_b = hf_state[f'{hf_prefix}.self_attn.q_proj.bias']
            k_b = hf_state[f'{hf_prefix}.self_attn.k_proj.bias']
            v_b = hf_state[f'{hf_prefix}.self_attn.v_proj.bias']
            openai_state[f'{oa_prefix}.attn.in_proj_bias'] = torch.cat([q_b, k_b, v_b], dim=0)

        if f'{hf_prefix}.self_attn.out_proj.weight' in hf_state:
            openai_state[f'{oa_prefix}.attn.out_proj.weight'] = hf_state[f'{hf_prefix}.self_attn.out_proj.weight']
        if f'{hf_prefix}.self_attn.out_proj.bias' in hf_state:
            openai_state[f'{oa_prefix}.attn.out_proj.bias'] = hf_state[f'{hf_prefix}.self_attn.out_proj.bias']

        # layer_norm1 -> ln_1
        if f'{hf_prefix}.layer_norm1.weight' in hf_state:
            openai_state[f'{oa_prefix}.ln_1.weight'] = hf_state[f'{hf_prefix}.layer_norm1.weight']
        if f'{hf_prefix}.layer_norm1.bias' in hf_state:
            openai_state[f'{oa_prefix}.ln_1.bias'] = hf_state[f'{hf_prefix}.layer_norm1.bias']

        # mlp
        if f'{hf_prefix}.mlp.fc1.weight' in hf_state:
            openai_state[f'{oa_prefix}.mlp.c_fc.weight'] = hf_state[f'{hf_prefix}.mlp.fc1.weight']
        if f'{hf_prefix}.mlp.fc1.bias' in hf_state:
            openai_state[f'{oa_prefix}.mlp.c_fc.bias'] = hf_state[f'{hf_prefix}.mlp.fc1.bias']
        if f'{hf_prefix}.mlp.fc2.weight' in hf_state:
            openai_state[f'{oa_prefix}.mlp.c_proj.weight'] = hf_state[f'{hf_prefix}.mlp.fc2.weight']
        if f'{hf_prefix}.mlp.fc2.bias' in hf_state:
            openai_state[f'{oa_prefix}.mlp.c_proj.bias'] = hf_state[f'{hf_prefix}.mlp.fc2.bias']

        if f'{hf_prefix}.layer_norm2.weight' in hf_state:
            openai_state[f'{oa_prefix}.ln_2.weight'] = hf_state[f'{hf_prefix}.layer_norm2.weight']
        if f'{hf_prefix}.layer_norm2.bias' in hf_state:
            openai_state[f'{oa_prefix}.ln_2.bias'] = hf_state[f'{hf_prefix}.layer_norm2.bias']

    # ln_final
    if 'text_model.final_layer_norm.weight' in hf_state:
        openai_state['ln_final.weight'] = hf_state['text_model.final_layer_norm.weight']
    if 'text_model.final_layer_norm.bias' in hf_state:
        openai_state['ln_final.bias'] = hf_state['text_model.final_layer_norm.bias']

    # text_projection: OpenAI 的 text_projection 是 [d_model, embed_dim]
    # HF 的 text_projection.weight 是 [embed_dim, d_model], 需要转置
    if 'text_projection.weight' in hf_state:
        openai_state['text_projection'] = hf_state['text_projection.weight'].T

    return openai_state


def main():
    if len(sys.argv) != 3:
        print("用法: python3 convert_clip_weights.py <hf_weight.pt> <output.pt>")
        sys.exit(1)

    hf_path = sys.argv[1]
    out_path = sys.argv[2]

    print(f"Loading HF weights from {hf_path}...")
    hf_state = torch.load(hf_path, map_location='cpu')

    if 'state_dict' in hf_state:
        hf_state = hf_state['state_dict']

    print(f"HF keys: {len(hf_state)}")
    print("Converting to OpenAI format...")
    openai_state = convert_hf_to_openai(hf_state)
    print(f"OpenAI keys: {len(openai_state)}")

    # 打印一些关键 key 确认
    print("\n关键 key 确认:")
    for k in ['visual.conv1.weight', 'visual.positional_embedding', 'visual.proj',
              'token_embedding.weight', 'positional_embedding', 'ln_final.weight',
              'text_projection', 'logit_scale']:
        if k in openai_state:
            print(f"  {k}: {openai_state[k].shape}")
        else:
            print(f"  {k}: MISSING!")

    print(f"\nSaving to {out_path}...")
    torch.save(openai_state, out_path)
    print(f"Done! Size: {os.path.getsize(out_path) / 1e9:.2f} GB")


if __name__ == '__main__':
    main()
