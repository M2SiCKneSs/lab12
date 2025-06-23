from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import WaveformGenerator, FileStream
from skmultiflow.trees import HoeffdingTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.drift_detection import DDM, EDDM, ADWIN
import matplotlib


# # 1. Create a synthetic stream
# stream = WaveformGenerator()  # 21 dims, 3 classes 
# clf = HoeffdingTreeClassifier()  
# evaluator = EvaluatePrequential(
#     show_plot=True,
#     pretrain_size=200,
#     max_samples=20000
# )  
# evaluator.evaluate(stream=stream, model=clf)


# from skmultiflow.drift_detection import DDM
# import numpy as np

# # Simulate correct (1) / incorrect (0) predictions
# stream = [1]*50 + [0]*50 + [1]*50
# ddm = DDM()  # using default warning=2.0, out_control=3.0 :contentReference[oaicite:19]{index=19}

# for i, correct in enumerate(stream):
#     ddm.add_element(int(correct))
#     if ddm.detected_change():
#         print(f"Drift detected at sample {i}")
#     elif ddm.detected_warning_zone():
#         print(f"Warning zone at sample {i}")


# electricity-market stream (45 312 rows, 8 features, binary target)
stream = FileStream('elec.csv', target_idx=-1, n_targets=1)
stream.prepare_for_use()
print("Features:", stream.n_features, "  Classes:", stream.target_values)

ht  = HoeffdingTreeClassifier()                 # single VFDT tree 
arf = AdaptiveRandomForestClassifier(           # ensemble with per-tree ADWIN 
        n_estimators=10, max_features='sqrt')

evaluator = EvaluatePrequential(
    pretrain_size = 200,
    n_wait        = 1_000,
    max_samples   = 6_000,
    metrics       = ['accuracy', 'kappa', 'precision', 'recall'],
    show_plot     = True)                       # will open a live Matplotlib window

evaluator.evaluate(
    stream       = stream,
    model        = [ht, arf],
    model_names  = ['VFDT', 'ARF'])



means_ht  = evaluator.get_mean_measurements(model_idx=0)
means_arf = evaluator.get_mean_measurements(model_idx=1)


# 1. re-run evaluation with show_plot=False and keep a manual log
stream.restart()
ht = HoeffdingTreeClassifier()
acc_hist = []

for i in tqdm(range(stream.n_remaining_samples())):
    X, y = stream.next_sample()
    y_pred = ht.predict(X)
    acc_hist.append(int(y_pred[0] == y[0]))
    ht.partial_fit(X, y, classes=[0, 1])

# 2. cumulative accuracy
cum_acc = np.cumsum(acc_hist) / np.arange(1, len(acc_hist)+1)

plt.figure()
plt.plot(cum_acc)
plt.xlabel('Samples processed')
plt.ylabel('Cumulative accuracy')
plt.title('Hoeffding Tree on Elec2')
plt.show()


stream.restart()
model     = HoeffdingTreeClassifier()
detector  = DDM(warning_level=2.5, out_control_level=4.0)
hits      = 0

for i in range(stream.n_remaining_samples()):
    X, y = stream.next_sample()
    pred = model.predict(X)[0]
    hits += (pred == y[0])

    detector.add_element(int(pred != y[0]))  # 1 = error for DDM/EDDM
    if hasattr(detector, 'detected_change') and detector.detected_change():
        print(f'drift at {i}; model reset')
        model = HoeffdingTreeClassifier()
        detector.reset()

    model.partial_fit(X, y, classes=[0, 1])

print(f' accuracy: {hits / 45_312:.4f}')


# After any EvaluatePrequential run
print("\n=== Evaluation summary ===")
print(evaluator.evaluation_summary())     # pretty, human-readable

