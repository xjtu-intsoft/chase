(import 'duorat-base.libsonnet')(output_from=true) {
    lr_scheduler: {
        "decay_steps": 98000,
        "end_lr": 0,
        "name": "warmup_polynomial",
        "num_warmup_steps": 2000,
        "power": 1,
        "start_lr": 0.0005
    },
    model+: {
        name: 'DuoRAT',
        encoder: {
            initial_encoder: {
                name: 'Bert',
                pretrained_model_name_or_path: 'bert-base-uncased',
                trainable: false,
                num_return_layers: 1,
                embed_dim: 256,
                use_dedicated_gpu: false,
                use_affine_transformation: true,
                use_attention_mask: false,
                use_token_type_ids: false,
                use_position_ids: false,
                use_segments: true
            },
            "rat_attention_dropout": 0.1,
            "rat_dropout": 0.1,
            "rat_ffn_dim": 256,
            "rat_num_heads": 8,
            "rat_num_layers": 6,
            "rat_relu_dropout": 0.1,
            source_relation_types: {
                use_schema_linking: true,
            },
        },
        decoder: {
            "action_embed_dim": 64,
            "field_embed_dim": 64,
            "type_embed_dim": 64,
            "p_mask": 0.2,
            "rat_attention_dropout": 0.1,
            "rat_dropout": 0.1,
            "rat_ffn_dim": 256,
            "rat_num_heads": 8,
            "rat_num_layers": 2,
            "rat_relu_dropout": 0.1
        },
        preproc+: {
            name: 'BertDuoRAT',
            add_cls_token: true,
            add_sep_token: false,

            min_freq: 5,
            max_count: 5000,

            tokenizer: {
                name: 'BERTTokenizer',
                pretrained_model_name_or_path: 'bert-base-uncased',
            },
        },
    },
    "train": {
        "amp_enabled": true,
        "batch_size": 20,
        "eval_batch_size": 20,
        "eval_beam_size": 1,
        "eval_decode_max_time_step": 500,
        "eval_every_n": 5000,
        "eval_nproc": 1,
        "eval_on_train": false,
        "eval_on_val": true,
        "infer_min_n": 5000,
        "max_steps": 100000,
        "n_grad_accumulation_steps": 12,
        "num_eval_items": 1034,
        "report_every_n": 10
    }
}
