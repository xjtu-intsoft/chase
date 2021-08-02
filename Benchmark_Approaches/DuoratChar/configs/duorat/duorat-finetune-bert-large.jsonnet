(import 'duorat-base.libsonnet')(output_from=true) {
    lr_scheduler: {
        "decay_steps": 98000,
        "end_lr": 0,
        "name": "bert_warmup_polynomial",
        "num_warmup_steps": 2000,
        "power": 1,
        "start_lr": 0.0001,
        "bert_factor": 8
    },
    model+: {
        name: 'LMDuoRAT',
        encoder: {
            initial_encoder: {
                name: 'Bert',
                pretrained_model_name_or_path: 'bert-base-chinese',
                trainable: true,
                num_return_layers: 1,
                embed_dim: 256,
                use_dedicated_gpu: false,
                use_affine_transformation: false,
                use_attention_mask: false,
                use_token_type_ids: false,
                use_position_ids: false,
                use_segments: false
            },
            "rat_attention_dropout": 0.1,
            "rat_dropout": 0.1,
            "rat_ffn_dim": 1024,
            "rat_num_heads": 8,
            "rat_num_layers": 8,
            "rat_relu_dropout": 0.1,
            source_relation_types: {
                use_schema_linking: true,
            },
            schema_input_token_ordering: '[column][table]',
            schema_source_token_ordering: '[column][table]',
            max_source_length: 200,
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
            "rat_relu_dropout": 0.1,
            pointer: {
                name: 'BahdanauMemEfficient',
                proj_size: 50,
            },
        },
        preproc+: {
            save_path: 'data/duorat-bert-large',
            name: 'BertDuoRAT',
            add_cls_token: true,
            add_sep_token: false,

            min_freq: 5,
            max_count: 5000,

            tokenizer: {
                name: 'BERTTokenizer',
                pretrained_model_name_or_path: 'bert-base-chinese',
            },
            transition_system+: {
                tokenizer: {
                    name: 'BERTTokenizer',
                    pretrained_model_name_or_path: 'bert-base-chinese',
                }
            }
        },
    },
    "train": {
        "amp_enabled": true,
        "batch_size": 12,
        "n_grad_accumulation_steps": 2,
        "eval_batch_size": 20,
        "eval_beam_size": 1,
        "eval_decode_max_time_step": 500,
        "eval_every_n": 5000,
        "eval_nproc": 1,
        "eval_on_train": false,
        "eval_on_val": true,
        "infer_min_n": 5000,
        "max_steps": 100000,
        "num_eval_items": 1034,
        "report_every_n": 10
    }
}
