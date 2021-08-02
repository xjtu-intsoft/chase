function(output_from, data_path='data/', save_dir='duorat/') {
    local PREFIX = data_path,

    data: {
        train: (import '../../data/train.libsonnet')(prefix=PREFIX),
        val: (import '../../data/val.libsonnet')(prefix=PREFIX),
    },

    model: {
        name: 'DuoRAT',
        encoder: {
            initial_encoder: {
                name: 'Transformer',
                num_heads: 2,
                ffn_dim: 128,
                dropout: 0.1,
                num_layers: 2,
                use_attention_mask: true,
                use_position_ids: true,
                use_positional_embedding: true,
            },
            rat_num_heads: 2,
            rat_ffn_dim: 128,
            rat_dropout: 0.1,
            rat_attention_dropout: 0.1,
            rat_relu_dropout: 0.1,
            rat_num_layers: 2,
            input_attention_scoping: {
                name: 'FineScoping',
                question_sees_columns: false,
                question_sees_tables: false,
                columns_see_question: false,
                columns_see_each_other: false,
                columns_see_tables: false,
                tables_see_question: false,
                tables_see_columns: false,
                tables_see_each_other: false,
                target_sees_question: false,
                target_sees_columns: false,
                target_sees_tables: false,
            },
            source_attention_scoping: {
                name: 'NoScoping',
            },
            source_relation_types: {
                use_schema_linking: true,
                high_confidence_db_content_schema_linking: true,
                low_confidence_db_content_schema_linking: false,
            },
            schema_input_token_ordering: '[column][table]',
            schema_source_token_ordering: '[column][table]',
        },
        decoder: {
            action_embed_dim: 64,
            field_embed_dim: 64,
            type_embed_dim: 64,
            rat_num_heads: 2,
            rat_ffn_dim: 128,
            rat_dropout: 0.1,
            rat_attention_dropout: 0.1,
            rat_relu_dropout: 0.1,
            rat_num_layers: 2,
            p_mask: 0.1,
            pointer: {
                name: 'Bahdanau',
                proj_size: 50,
            },
            target_attention_scoping: {
                name: 'NoScoping'
            },
            target_relation_types: {},
            memory_relation_types: {
                copied_from_relation: true,
            },
            grammar_constrained_training: true,
            grammar_constrained_inference: true,
        },
        preproc: {
            name: 'TransformerDuoRAT',
            min_freq: 5,
            max_count: 5000,
            use_full_glove_vocab: false,
            train_num_schema_shuffles: 0,
            val_num_schema_shuffles: 0,
            save_path: PREFIX + save_dir,
            tokenizer: {
                name: 'CoreNLPTokenizer',
            },
            transition_system: {
                name: 'SpiderTransitionSystem',
                asdl_grammar_path: 'duorat/asdl/lang/spider/spider_asdl.txt',
                tokenizer: {
                    name: 'CoreNLPTokenizer',
                },
                output_from: output_from,
                use_table_pointer: output_from,
                include_literals: true,
                include_columns: true,
            },
            schema_linker: {
                name: 'SpiderSchemaLinker',
                tokenizer: {
                    name: 'CoreNLPTokenizer',
                },
                max_n_gram: 10,
                with_stemming: false,
                whole_entry_db_content_confidence: 'none',
                partial_entry_db_content_confidence: 'high'
            },
        },
    },

    train: {
        amp_enabled: false,
        batch_size: 10,
        n_grad_accumulation_steps: 4,
        eval_batch_size: 50,
        eval_on_train: false,
        eval_on_val: true,
        eval_nproc: 1,
        eval_beam_size: 1,
        eval_decode_max_time_step: 500,

        eval_every_n: 1000,
        report_every_n: 10,
        infer_min_n: 1000,

        max_steps: 80000,
        num_eval_items: 1034,
    },
    optimizer: {
        name: 'adam',
        lr: 0.0,
    },
    lr_scheduler: {
        name: 'warmup_polynomial',
        num_warmup_steps: $.train.max_steps / 20,
        start_lr: 1e-3,
        end_lr: 0,
        decay_steps: $.train.max_steps - self.num_warmup_steps,
        power: 0.5,
    }
}
