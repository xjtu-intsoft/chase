(import 'duorat-base.libsonnet')(output_from=true) {
    model+: {
        encoder+: {
            initial_encoder+: {
                num_heads: 6,
                ffn_dim: 256,
                dropout: 0.1,
                num_layers: 2,
                use_attention_mask: true,
                use_position_ids: true,
                use_positional_embedding: true,
            },
            rat_attention_dropout: 0.1,
            rat_dropout: 0.1,
            rat_ffn_dim: 256,
            rat_num_heads: 10,
            rat_num_layers: 6,
            rat_relu_dropout: 0.1,
            source_relation_types: {
                use_schema_linking: true,
                only_exact_matches: false
            },
        },
        decoder+: {
            rat_num_heads: 8,
            rat_ffn_dim: 256,
            p_mask: 0.2,
        },
    },
    train+: {
        batch_size: 20,
        n_grad_accumulation_steps: 12,
        eval_batch_size: 50,
        eval_every_n: 5000,
        infer_min_n: 5000,
        max_steps: 100000,
    },
    lr_scheduler+: {
        num_warmup_steps: 2000,
        decay_steps: 98000,
        power: 1,
        start_lr: 0.0005,
    },
}
