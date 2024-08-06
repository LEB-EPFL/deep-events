from pathlib import Path
import os
import re
import csv
from typing import Callable

import tifffile
import tensorflow as tf
import numpy as np
from benedict import benedict
import deep_events.training_functions
from deep_events.database.convenience import get_latest_folder, get_latest
from deep_events.train import adjust_tf_dimensions
from mitosplit_net import evaluation
from skimage.filters import threshold_otsu

FOLDER = Path("W:/deep_events/data/original_data/training_data")
#w_tp=1, w_tn=1, w_fp=1, w_fn=1
# MCC_WEIGHTS = [5, 1, 15, 1]
# MCC_WEIGHTS = [5, 1, 5, 1]
# MCC_WEIGHTS = [5, 3, 10, 1]
MCC_WEIGHTS = [5, 5, 10, 1]

def whole_ev_eval(eval_mask, pred_output_test, eval_images, plot, eval_frames, threshold=None):
    event_values = []
    for event_frames in eval_frames:
        event_values.append(pred_output_test[event_frames[0]:event_frames[-1] + 1].max())

    if threshold is None:
        threshold = max(threshold_otsu(np.array(event_values)), 0.5)

    TP, FP, FN, TN = 0, 0, 0, 0
    tp_values, fp_values, fn_values, tn_values = [], [], [], []
    # threshold = .5
    for event_frames in eval_frames:
        max_pred = pred_output_test[event_frames[0]:event_frames[-1] + 1].max()
        max_gt = eval_mask[event_frames[0]:event_frames[-1] + 1].max() 
        if max_gt > threshold:
            if max_pred > threshold:
                TP += 1
                tp_values.append(max_pred)
            if max_pred < threshold:
                FN += 1
                fn_values.append(max_pred)
        else:
            if max_pred > threshold:
                FP += 1
                fp_values.append(max_pred)
            if max_pred < threshold:
                TN += 1
                tn_values.append(max_pred)

    return [TP, FP, FN, TN, tp_values, fp_values, fn_values, tn_values, threshold]

def event_loc_eval(eval_mask, pred_output_test, eval_images, plot, *_):
    labels = evaluation.label(pred_output_test, 0.7)
    labels = np.expand_dims(labels, axis=-1)
    true_labels = evaluation.label(eval_mask, 0.7)

    if plot:
        try:
            import matplotlib.pyplot as plt
            import random
            to_plot = random.sample(range(labels.shape[0]), labels.shape[0])
            frame = 0
            while True:
                f, axs = plt.subplots(1, 3)
                f.set_size_inches(25,10)
                axs[0].imshow(labels[to_plot[frame], :, :, 0], vmax=1, cmap='Reds')
                axs[0].set_title("prediction")
                axs[1].imshow(true_labels[to_plot[frame], :, :, 0])
                axs[1].set_title("ground truth")
                axs[2].imshow(eval_images[to_plot[frame], :, :, 0], cmap='gray')
                frame += 1
                plt.show()
        except KeyboardInterrupt:
            pass

    return evaluation.fissionStatsStack(true_labels, labels)


def main(model_dir: Path|str = None, write_yaml: bool = True, plot = False, general_eval: bool = False,
         eval_func: Callable = whole_ev_eval, save_hist: bool = True):
    no_details=False
    model_dir = Path(model_dir)
    if model_dir is None:
        training_folder = Path(get_latest_folder(FOLDER)[0])
        model_dir = Path(get_latest("model", training_folder))
        print(model_dir)
    elif str(model_dir)[-3:] != '.h5':
        training_folder = model_dir
        model_dir = list(training_folder.glob('*model.h5'))[0]
        print(model_dir)
    else:
        training_folder = model_dir.parent

    if general_eval:
        parts = model_dir.parent.name.split("_")[2:]
        # Remove subset, so that we check on full validation data
        subset = [re.match("s0\..*", x) for x in parts]
        parts = [x for x, sub in zip(parts, subset) if not sub]
        pattern = "*" + "_".join(parts)

        print(pattern)
        print(model_dir.parents[1])
        training_folder = Path(get_latest_folder(model_dir.parents[1], pattern)[0])
        print(training_folder)

    model = tf.keras.models.load_model(model_dir, compile=False)

    eval_images = adjust_tf_dimensions(tifffile.imread(training_folder / "eval_images_00.tif"))
    frames = eval_images.shape[0]
    eval_images = eval_images[:frames]

    try:
        with open(training_folder / 'eval_events.csv', 'r') as read_obj: 
            csv_reader = csv.reader(read_obj) 
            eval_event_frames = list(csv_reader)
            for i in range(len(eval_event_frames)):
                eval_event_frames[i] = [int(x) for x in eval_event_frames[i]]
    except FileNotFoundError:
        eval_func = event_loc_eval
        eval_event_frames = []
        no_details = True

    eval_mask = adjust_tf_dimensions(tifffile.imread(training_folder / "eval_gt_00.tif"))[:frames]
    pred_output_test = evaluation.predict(eval_images, model)

    # [TP, FP, FN, TP_px, FP_px, FN_px]
    # [TP, FP, FN, TN, tp_values, fp_values, fn_values, tn_values, threshold] for whole_event
    threshold = optimize_threshold(eval_mask, pred_output_test, eval_images, model_dir, eval_event_frames)
    stats = eval_func(eval_mask, pred_output_test, eval_images, plot, eval_event_frames, threshold)
    print(stats)
    precision = evaluation.get_precision(stats[0], stats[1])
    recall = evaluation.get_tpr(stats[0], stats[2])
    precision = round(precision*100)/100
    recall = round(recall*100)/100
    try:
        f1 = round(evaluation.get_f1_score(precision, recall)*100)/100
    except (ValueError, TypeError) as e:
        print(precision)
        print(recall)
        import traceback
        traceback.print_exc()
        f1 = 0
    summary = f"""
    {model_dir}
    precision {precision}
    recall {recall}
    f1 {f1}
    """
    if not no_details:
        mcc = evaluation.get_mcc(*stats[:4])
        kappa = evaluation.get_kappa(*stats[:4])
        mcc = round(mcc*100)/100
        kappa = round(kappa*100)/100
        w_mcc = evaluation.get_weighted_mcc(*stats[:4], *MCC_WEIGHTS)
        summary = summary + f"mcc {mcc}\n    w_mcc {w_mcc}\n    kappa {kappa}"
    
    print(summary)

    if write_yaml:
        settings = benedict(str(model_dir).replace("model.h5", "settings.yaml"))
        settings["performance"] = {}
        try:
            settings["performance"]["threshold"] = stats[8]
        except:
            pass
        settings["performance"]["precision"] = precision
        settings["performance"]["recall"] = recall
        settings["performance"]["f1"] = f1
        if not no_details:
            settings["performance"]["mcc"] = mcc
            settings["performance"]["w_mcc"] = w_mcc
            settings["performance"]["w_mcc_weights"] = {"tp":MCC_WEIGHTS[0], "tn":MCC_WEIGHTS[1],
                                                        "fp":MCC_WEIGHTS[2], "fn":MCC_WEIGHTS[3]}
            settings["performance"]["kappa"] = kappa
        settings["eval_data"] = training_folder
        settings["performance"]["eval"] = "v1" if no_details else "v2"
        settings.to_yaml(filepath=str(model_dir).replace("model.h5", "settings.yaml"))
    if save_hist and not no_details:
        import matplotlib.pyplot as plt
        bins = np.arange(0., 1.001, .05)
        plt.figure()
        plt.hist(stats[4], alpha=.5, label="tp_values", bins=bins, color="green", edgecolor="green", linewidth=2)
        plt.hist(stats[5], alpha=.5, label="fp_values", bins=bins, color="red", edgecolor="red", linewidth=2)
        plt.hist(stats[6], alpha=.5, label="fn_values", bins=bins, color="red", edgecolor="green", linewidth=2)
        plt.hist(stats[7], alpha=.5, label="tn_values", bins=bins, color="green", edgecolor="red", linewidth=2)
        plt.axvline(x=stats[8])
        plt.legend(loc='upper right')
        plt.savefig(str(model_dir).replace("model.h5", "hist.png"))
        plt.close()

def performance_for_folder(folder:Path, general_eval=True, older_than=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    models = folder.rglob("*_model.h5")
    for model in models:
        if int(model.parts[-1][:len(str(older_than))]) < older_than:
            continue
        print('\033[1m' + str(model) + '\033[0m')
        main(model, general_eval=general_eval)

def optimize_threshold(eval_mask, pred_output_test, eval_images, plot, eval_event_frames):
    results = {"thresholds": [], "precision": [], "recall": [], "f1": [], "mcc": [], "kappa": [],
           "specificity": [], "npv": [], "w_mcc": []}
    for threshold in np.linspace(0, 1, 101):
        stats = whole_ev_eval(eval_mask, pred_output_test, eval_images, False, eval_event_frames, threshold)
        
        try:
            precision = evaluation.get_precision(stats[0], stats[1])
        except:
            precision = 0
        try:
            recall = evaluation.get_tpr(stats[0], stats[2])
        except:
            recall = 0
        try:
            f1 = round(evaluation.get_f1_score(precision, recall)*100)/100
        except:
            f1 = 0
        specificity = round(stats[3] / (stats[3] + stats[1]) * 100)/100 if (stats[3] + stats[2]) != 0 else 0
        npv = round(stats[3] / (stats[3] + stats[2]) * 100)/100 if (stats[3] + stats[2]) != 0 else 0
        precision = round(precision*100)/100
        recall = round(recall*100)/100 
        mcc = evaluation.get_mcc(*stats[:4])
        #w_tp=1, w_tn=1, w_fp=1, w_fn=1

        w_mcc = evaluation.get_weighted_mcc(*stats[:4], *MCC_WEIGHTS)
        kappa = evaluation.get_kappa(*stats[:4])
        mcc = round(mcc*100)/100
        kappa = round(kappa*100)/100
        results["thresholds"].append(threshold)
        results["precision"].append(precision)
        results["specificity"].append(specificity)
        results["npv"].append(npv)
        results["recall"].append(recall)
        results["f1"].append(f1)
        results["mcc"].append(mcc)
        results["w_mcc"].append(w_mcc)
        results["kappa"].append(kappa)
    wmcc_threshold = results["thresholds"][results["w_mcc"].index(max(results["w_mcc"]))]
    try:
        recall_threshold = max([x for i, x in enumerate(results["thresholds"]) if results["recall"][i] > 0.5])
    except ValueError:
        recall_threshold = 0
    print(wmcc_threshold, recall_threshold)
    threshold = min(wmcc_threshold, recall_threshold)
    # threshold = wmcc_threshold
    threshold = max(threshold, 0.5)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        scores = ["f1", "mcc", "w_mcc", "recall"]
        for score in scores:
            plt.plot(results["thresholds"], results[score])
        plt.axvline(wmcc_threshold)
        plt.axvline(threshold)
        plt.legend(scores)
        plt.savefig(str(plot).replace("model.h5", "perf.png"))
        plt.close()
    return threshold


def visual_eval(training_folder, model_name = None):
    import matplotlib.pyplot as plt
    frame = 1
    if model_name is None:
        model_dir = Path(get_latest("model", training_folder))
    else:
        model_dir = training_folder / model_name
    print(model_dir)
    model = tf.keras.models.load_model(model_dir)
    eval_images = adjust_tf_dimensions(tifffile.imread(training_folder / "eval_images_00.tif"))
    eval_mask = adjust_tf_dimensions(tifffile.imread(training_folder / "eval_gt_00.tif"))
    while True:
        output = model.predict(np.expand_dims(eval_images[frame],axis=0))
        f, axs = plt.subplots(1, 3)
        f.set_size_inches(15,5)
        axs[0].imshow(output[0, :, :, 0], vmax=1, cmap='Reds')
        axs[0].set_title("prediction")
        axs[1].imshow(eval_mask[frame, :, :, 0])
        axs[1].set_title("ground truth")
        axs[2].imshow(eval_images[frame, :, :, 0], cmap='gray')
        plt.show()
        frame = frame + 1

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#     main(Path("W:/deep_events/data/original_data/training_data/20230626_1508_brightfield_cos7/20230626_1509_model.h5"))
    # main(Path("W:/deep_events/data/original_data/training_data/20230626_1509_fluorescence_zeiss_cos7/20230626_1509_model.h5"))
    # main(Path("Z:/_Lab members/Juan/Experiments/230222_MitoSplitNet_TrainingSet_U2OS_iSIM/training_data/20230611_0201_isim_cos7/20230611_0202_model.h5"))
    folder = Path("W:/deep_events/data/original_data/training_data/20240704_0955_brightfield_cos7_n7_f1_s0.01")
    # folder = Path(r"Y:\SHARED\_Scientific projects\ADA_WS_JCL\Phase_PDA\training_data\20231126_1914_zeiss_s1_iFalse_mitochondria") # "Z:/SHARED/_Scientific projects/ADA_WS_JCL/Phase_PDA/training_data/20231121_1606_zeiss_s1_iFalse_mitochondria")
    model = "20240704_0955_model" + ".h5"
    visual_eval(folder, model)
    # visual_eval(Path("W:/deep_events/data/original_data/training_data/20230718_0123_brightfield_cos7"),
    #             "20230718_0128_model.h5")