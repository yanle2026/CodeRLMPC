from PredictorLSTMKeras import LSTMNoisePredictor

# 示例使用
predictor = LSTMNoisePredictor(mean=[0, 0], std_dev=[0.1, 0.1], lower=[-1, -1], upper=[1, 1],
                                timesteps=10, num_samples=1000,
                                input_size=2, hidden_size=64, output_size=2, num_layers=2)

predictor.train(num_epochs=10)
y_pred_rescaled, y_true_rescaled = predictor.evaluate()
predictor.plot_results(y_pred_rescaled, y_true_rescaled)