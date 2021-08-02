(import 'duorat-good-no-bert.jsonnet') {
    train+: {
        batch_size: 8,
        n_grad_accumulation_steps: 12,
    },
}