(import 'duorat-finetune-bert-large.jsonnet') {
    model+: {
        encoder+: {
            initial_encoder+: {
                use_attention_mask: true,
            },
            input_attention_scoping: {
                name: 'CoarseScoping',
                question_sees_schema: true,
                schema_sees_question: true,
            },
            source_attention_scoping: {
                name: 'CoarseScoping',
                question_sees_schema: true,
                schema_sees_question: true,
                target_sees_question: true,
                target_sees_schema: true,
            },
        },
        decoder+: {
            pointer_proj_size: 50,
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
    },
}
