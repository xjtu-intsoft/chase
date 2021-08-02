(import 'duorat-finetune-bert-large.jsonnet') {
    model+: {
        encoder+: {
            source_relation_types: {
                use_schema_linking: true,
                high_confidence_db_content_schema_linking: false,
                low_confidence_db_content_schema_linking: true,
            }
        },
        preproc+: {
            save_path: 'data/duorat/duorat-bert-large-wn-pl',
            schema_linker+: {
                whole_entry_db_content_confidence: 'none',
                partial_entry_db_content_confidence: 'low'
            }
        }
    },
}
