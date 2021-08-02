(import 'duorat-base.libsonnet')(output_from=true) {
    model+: {
        encoder+: {
            initial_encoder+: {
                num_heads: 1,
                ffn_dim: 32,
                num_layers: 1,
            },
            rat_num_heads: 1,
            rat_ffn_dim: 32,
        },
        decoder+: {
            action_embed_dim: 16,
            field_embed_dim: 16,
            type_embed_dim: 16,
            rat_num_heads: 1,
            rat_ffn_dim: 32,
        },
    },
    train+: {
        batch_size: 2
    },
}