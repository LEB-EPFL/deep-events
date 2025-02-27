from pathlib import Path
import os
import re
import csv
from typing import Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import tifffile
import tensorflow as tf
import numpy as np
from benedict import benedict
import deep_events.lstm_models
from deep_events.database.convenience import get_latest_folder, get_latest
from deep_events.train import adjust_tf_dimensions
from deep_events.generator import ArraySequence
from mitosplit_net import evaluation
from skimage.filters import threshold_otsu

FOLDER = Path("W:/deep_events/data/original_data/training_data")
SETTINGS_NAME = 'testing'
SETTINGS_NAME = 'f1_settings'
SETTINGS_NAME = 'settings'
# SETTINGS_NAME = 'frame'
#w_tp=1, w_tn=1, w_fp=1, w_fn=1
# MCC_WEIGHTS = [5, 1, 15, 1]
# MCC_WEIGHTS = [5, 1, 5, 1]
# MCC_WEIGHTS = [5, 3, 10, 1]
MCC_WEIGHTS = [5, 5, 10, 1]
beta = 0.1

def whole_ev_eval(masks, preds, plot, eval_frames, threshold=None):
    if threshold is None:
        threshold = threshold_otsu(np.array(preds))
    TP, FP, FN, TN = 0, 0, 0, 0
    tp_values, fp_values, fn_values, tn_values = [], [], [], []
    # threshold = .5
    for gt, pred in zip(masks, preds):
        # for gt_frame, pred_frame in zip(mask, pred):
        max_pred = pred.max()
        max_gt = gt.max() 
        if max_gt > 0.5:
            if max_pred > threshold:
                TP += 1
                tp_values.append(max_pred)
            else:
                FN += 1
                fn_values.append(max_pred)
        else:
            if max_pred > threshold:
                FP += 1
                fp_values.append(max_pred)
            else:
                TN += 1
                tn_values.append(max_pred)
    return [TP, FP, FN, TN, tp_values, fp_values, fn_values, tn_values, threshold]

def event_loc_fp_eval(eval_mask, pred_output_test, plot, eval_frames, threshold):
    all_preds = evaluation.label(pred_output_test, threshold) 
    all_true_labels = evaluation.label(eval_mask, 0.7)
    TP, FP, FN, TN = np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0)
    tp_values, fp_values, fn_values, tn_values = [], [], [], []
    for labels, preds, maps in zip(all_true_labels, all_preds, pred_output_test):
        for label, pred, map in zip(labels, preds[:, 0, :, :, 0], maps[:, 0, :, :, 0]):
            pred_copy = pred.copy()
            for event in np.unique(label)[1:]:
                pred_labels = pred[label==event]
                values = map[label==event]
                if pred_labels.max() > 0:
                    TP += 1
                    tp_values.append(values.max())
                    for tp in np.unique(pred_labels)[1:]:
                        # print('deleting TP', tp)
                        pred_copy[pred_copy == tp] = 0
                else:
                    FN += 1
                    fn_values.append(values.max())
            for event in np.unique(pred_copy)[1:]:
                FP += 1
                values = map[pred_copy==event]
                fp_values.append(values.max())
    # print(TP, FP, FN, TN, threshold)
    return [TP, FP, FN, TN, tp_values, fp_values, fn_values, tn_values, threshold] 

def frame_wise_eval(eval_mask, pred_output_test, plot, eval_frames, threshold):
    all_preds = evaluation.label(pred_output_test, threshold) 
    all_true_labels = evaluation.label(eval_mask, 0.5)
    TP, FP, FN, TN = np.uint32(0), np.uint32(0), np.uint32(0), np.uint32(0)
    tp_values, fp_values, fn_values, tn_values = [], [], [], []
    for labels, preds, maps in zip(all_true_labels, all_preds, pred_output_test):
        for label, pred, map in zip(labels, preds[:, 0, :, :, 0], maps[:, 0, :, :, 0]):
            value = map.max()
            if label.max() > 0:
                if pred.max() > 0:
                    TP += 1
                    tp_values.append(value)
                else: 
                    FN += 1
                    fn_values.append(value)
            else:
                if pred.max() > 0:
                    FP += 1
                    fp_values.append(value)
                else:
                    TN += 1
                    tn_values.append(value)
    # print(TP, FP, FN, TN, threshold)
    return [TP, FP, FN, TN, tp_values, fp_values, fn_values, tn_values, threshold] 

def event_loc_eval(eval_mask, pred_output_test, plot, *_):
    labels = evaluation.label(pred_output_test, 0.7)
    labels = np.expand_dims(labels, axis=-1)
    true_labels = evaluation.label(eval_mask, 0.7)

    if plot:
        try:
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
                # axs[2].imshow(eval_images[to_plot[frame], :, :, 0], cmap='gray')
                frame += 1
                plt.show()
        except KeyboardInterrupt:
            pass

    return evaluation.fissionStatsStack(true_labels, labels[..., 0])

def pixel_wise_eval(eval_mask, pred_output_test, plot, eval_frames, threshold):
    predictions = pred_output_test[:, :, 0, : ,: ,0]
    ground_truth = eval_mask
    # print(predictions.shape)
    # print(eval_mask.shape)
    tolerance = 0.05
    mask = ground_truth > 0.05
    predictions = predictions[mask]
    ground_truth = ground_truth[mask]
    TP = int(np.sum(np.abs(predictions - ground_truth) <= tolerance))  # Matches within tolerance
    FP = int(np.sum((predictions > ground_truth + tolerance)))  # Predictions too high outside tolerance
    FN = int(np.sum((predictions < ground_truth - tolerance)))
    TN = int(predictions.size - (TP + FP + FN))  # Remaining are true negatives
    tp_values, fp_values, fn_values, tn_values = [], [], [], []
    return [TP, FP, FN, TN, tp_values, fp_values, fn_values, tn_values, threshold]


def main(model_dir: Path|str = None, write_yaml: bool = True, plot = False, general_eval: bool = False,
         eval_func: Callable = whole_ev_eval, save_hist: bool = True):
    no_details=True
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
    settings = benedict(str(model_dir).replace('model.h5', 'settings.yaml'))
    try:
        data_folder = Path(settings['data_folder'])
    except (NameError, KeyError):
        data_folder = training_folder
    try:
        frames = settings['n_timepoints']
    except KeyError:
        print(training_folder)
        if '_n' in str(training_folder):
            frames = int(str(training_folder).split('_n')[1][0])
        else:
            frames = 1
        print('n timepoints from folder name:', frames)

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

    model = tf.keras.models.load_model(model_dir, compile=False, custom_objects={'ChannelSlice': ChannelSlice})

    eval_seq = ArraySequence(data_folder, batch_size=1, n_augmentations=1, validation=True, t_size=frames)
    eval_seq.performance = True

    try:
        with open(data_folder / 'eval_events.csv', 'r') as read_obj: 
            csv_reader = csv.reader(read_obj) 
            eval_event_frames = list(csv_reader)
            for i in range(len(eval_event_frames)):
                eval_event_frames[i] = [int(x) for x in eval_event_frames[i]]
    except FileNotFoundError:
        eval_func = event_loc_eval
        eval_event_frames = []
        no_details = True
    n_repeats = 3

    masks = np.zeros((len(eval_seq)*n_repeats, 5, *eval_seq.__getitem__(i)[1].shape))
    images = np.zeros((len(eval_seq)*n_repeats, 5, *eval_seq.__getitem__(i)[0].shape))
    for i in tqdm(range((len(eval_seq)*n_repeats))):
        idx = i%len(eval_seq)
        for f in range(-5,0):
            eval_seq.last_frame = f
            image, mask = eval_seq.__getitem__(idx)
            masks[i, f] = mask
            images[i, f] = image
            # print(mask.max())
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(1, 3)
            # axs[0].imshow(image[0, :, :, 0])
            # axs[1].imshow(mask[0, :, :, 0])
            # axs[2].imshow(pred[-1][-1][0], clim=(0,1))
            # plt.show()
    print(images.shape)
    # masks, pred = np.asarray(masks), np.asarray(pred)
    images = images.reshape((len(eval_seq)*n_repeats*5, *images.shape[2:]))
    # images = np.moveaxis(images, 1, -1)
    images = images[:, 0, ...]
    print(images.shape)
    success = False
    parts = False
    while not success:
        try:
            if parts:
                pred = []
                for img in images:
                    img = np.expand_dims(img, 0)
                    pred.append(evaluation.predict(img, model))
                pred = np.vstack(pred)
            else:
                pred = evaluation.predict(images, model)
            success = True
        except Exception as e:
            print(e)
            parts = True
    print(pred.shape)
    pred = pred.reshape((len(eval_seq)*n_repeats, 5, *eval_seq.__getitem__(i)[1].shape))
    masks = np.squeeze(masks[:, :, 0, :, :])
    print(masks.shape)
    # [TP, FP, FN, TP_px, FP_px, FN_px]
    # [TP, FP, FN, TN, tp_values, fp_values, fn_values, tn_values, threshold] for whole_event
    threshold = optimize_threshold(masks, pred, model_dir, eval_event_frames, eval_func)
    # threshold = None
    stats = eval_func(masks, pred, False, eval_event_frames, threshold)
    print(stats[:4])
    precision = evaluation.get_precision(stats[0], stats[1])
    recall = evaluation.get_tpr(stats[0], stats[2])
    try:
        f1 = round(evaluation.get_f1_score(precision, recall)*100)/100
    except (ValueError, TypeError) as e:
        print(precision)
        print(recall)
        import traceback
        traceback.print_exc()
        f1 = 0
    if precision + recall == 0:
        fbeta = 0
    beta_squared = beta ** 2
    if precision + recall == 0:
        fbeta = 0
    else:
        fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
    fbeta = round(fbeta*100)/100
    precision = round(precision*100)/100
    recall = round(recall*100)/100
    summary = f"""
    {model_dir}
    precision {precision}
    recall {recall}
    f1 {f1}
    f_beta {fbeta}
    threshold {threshold}
    """
    if not no_details:
        mcc = evaluation.get_mcc(*stats[:4])
        kappa = evaluation.get_kappa(*stats[:4])
        mcc = round(mcc*100)/100
        kappa = round(kappa*100)/100
        w_mcc = evaluation.get_weighted_mcc(*stats[:4], *MCC_WEIGHTS)
        summary = summary + f"mcc {mcc}\n    w_mcc {w_mcc}\n    kappa {kappa}\n"
    
    print(summary)

    if write_yaml:
        settings = benedict(str(model_dir).replace("model.h5", f"settings.yaml"))
        settings["performance"] = {}
        try:
            settings["performance"]["threshold"] = stats[8]
        except:
            pass
        settings["performance"]["precision"] = precision
        settings["performance"]["recall"] = recall
        settings["performance"]["f1"] = f1
        settings["performance"]["fbeta"] = fbeta
        settings["performance"]["beta"] = beta
        if not no_details:
            settings["performance"]["mcc"] = mcc
            settings["performance"]["w_mcc"] = w_mcc
            settings["performance"]["w_mcc_weights"] = {"tp":MCC_WEIGHTS[0], "tn":MCC_WEIGHTS[1],
                                                        "fp":MCC_WEIGHTS[2], "fn":MCC_WEIGHTS[3]}
            settings["performance"]["kappa"] = kappa
        settings["eval_data"] = training_folder
        settings["performance"]["eval"] = "v1" if no_details else "v2"
        settings.to_yaml(filepath=str(model_dir).replace("model.h5", f"{SETTINGS_NAME}.yaml"))
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
        print(str(model_dir).replace("model.h5", "hist_data.yaml"))
        benedict({'pos': stats[4] + stats[6], 
                  'neg': stats[5] + stats[7]}).to_yaml(filepath=str(model_dir).replace("model.h5", "hist_data_local.yaml"))
    return f1


def performance_for_folder(folder:Path, general_eval=True, older_than=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    models = folder.rglob("*_model.h5")
    for model in models:
        if int(model.parts[-1][:len(str(older_than))]) < older_than:
            continue
        print('\033[1m' + str(model) + '\033[0m')
        main(model, general_eval=general_eval)

def optimize_threshold(eval_mask, pred_output_test, plot, eval_event_frames, eval_func=whole_ev_eval):
    results = {"thresholds": [], "precision": [], "recall": [], "f1": [], "mcc": [], "kappa": [],
           "specificity": [], "npv": [], "w_mcc": [], f"f_beta": [], 'beta':beta}
    for threshold in tqdm(np.linspace(0.05, 1, 96)):
        stats = eval_func(eval_mask, pred_output_test, False, eval_event_frames, threshold)
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
        specificity = round(stats[3] / (stats[3] + stats[1]) * 100)/100 if (stats[3] + stats[1]) != 0 else 0
        npv = round(stats[3] / (stats[3] + stats[2]) * 100)/100 if (stats[3] + stats[2]) != 0 else 0
        beta_squared = beta ** 2
        if precision + recall == 0:
            fbeta = 0
        else:
            fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
        precision = round(precision*100)/100
        recall = round(recall*100)/100 
        # mcc = evaluation.get_mcc(*stats[:4])
        mcc = 0
        #w_tp=1, w_tn=1, w_fp=1, w_fn=1

        # w_mcc = evaluation.get_weighted_mcc(*stats[:4], *MCC_WEIGHTS)
        # kappa = evaluation.get_kappa(*stats[:4])
        w_mcc = 0
        kappa = 0
        mcc = round(mcc*100)/100
        kappa = round(kappa*100)/100
        fbeta = round(fbeta*100)/100
        results[f"f_beta"].append(fbeta)
        results["thresholds"].append(threshold)
        results["precision"].append(precision)
        results["specificity"].append(specificity)
        results["npv"].append(npv)
        results["recall"].append(recall)
        results["f1"].append(f1)
        results["mcc"].append(mcc)
        results["w_mcc"].append(w_mcc)
        results["kappa"].append(kappa)
        # print(f'prec {precision}, w_mcc {w_mcc}, mcc {mcc}, recall {recall}')
    # wmcc_threshold = results["thresholds"][results["w_mcc"].index(max(results["w_mcc"]))]
    # threshold = wmcc_threshold
    # try:
    #     recall_threshold = max([x for i, x in enumerate(results["thresholds"]) if results["recall"][i] > 0.5])
    # except ValueError:
    #     recall_threshold = 0   # print(wmcc_threshold, recall_threshold)
    # threshold = min(wmcc_threshold, recall_threshold)
    fbeta_threshold = results["thresholds"][results[f"f_beta"].index(max(results[f"f_beta"]))]
    try:
        idx = next(x[0] for x in enumerate(results[f"precision"]) if x[1] > 0.98)
    except StopIteration:
        idx = results['precision'].index(max(results['precision']))

    f1_threshold =  results["thresholds"][results[f"f1"].index(max(results[f"f1"]))]
    threshold = fbeta_threshold
    # threshold = max(threshold, 0.5)
    # TODO reset threshold
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        scores = ["f1", "mcc", "w_mcc", "recall", "precision", f"f_beta"]
        for score in scores:
            plt.plot(results["thresholds"], results[score])
        plt.axvline(threshold)
        # plt.axvline(threshold)
        plt.legend(scores)
        plt.savefig(str(plot).replace("model.h5", "perf.png"))
        plt.close()
    benedict(results).to_yaml(filepath=str(plot).replace('model.h5', f'perf_data_{SETTINGS_NAME}.yaml'))
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

from tensorflow.keras.layers import Layer
class ChannelSlice(Layer):
    def __init__(self, idx, **kwargs):
        super(ChannelSlice, self).__init__(**kwargs)
        self.idx = idx

    def call(self, inputs):
        # Slices out the channel at index self.idx and keeps the dimension.
        return inputs[..., self.idx:self.idx+1]

    def get_config(self):
        config = super(ChannelSlice, self).get_config()
        config.update({'idx': self.idx})
        return config

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#     main(Path("W:/deep_events/data/original_data/training_data/20230626_1508_brightfield_cos7/20230626_1509_model.h5"))
    # main(Path("W:/deep_events/data/original_data/training_data/20230626_1509_fluorescence_zeiss_cos7/20230626_1509_model.h5"))
    # main(Path("Z:/_Lab members/Juan/Experiments/230222_MitoSplitNet_TrainingSet_U2OS_iSIM/training_data/20230611_0201_isim_cos7/20230611_0202_model.h5"))
    #### MITOS
    # main_folder = Path("x:/Scientific_projects/deep_events_WS/data/original_data/training_data/")
    # dirs = []
    # # LSTM focal
    # dirs.extend(list(main_folder.glob('20250106_*'))) 
    # dirs.extend(list(main_folder.glob('20250107_*')))
    # # UNET and focal
    # dirs.extend(list(main_folder.glob('20250109_*'))) 
    # dirs.extend(list(main_folder.glob('20250110_*'))) 
    # # UNET BCE
    # dirs.extend(list(main_folder.glob('20250113_*'))) 
    # dirs.extend(list(main_folder.glob('20250114_*'))) 
    # # LSTM BCE
    # dirs.extend(list(main_folder.glob('20250115_*')))
    # dirs.extend(list(main_folder.glob('20250116_*')))



    # results = {}
    # for key in dirs:
    #     results[key] = []

    # # folder = Path(r"Y:\SHARED\_Scientific projects\ADA_WS_JCL\Phase_PDA\training_data\20231126_1914_zeiss_s1_iFalse_mitochondria") # "Z:/SHARED/_Scientific projects/ADA_WS_JCL/Phase_PDA/training_data/20231121_1606_zeiss_s1_iFalse_mitochondria")
    # for folder in dirs:

    #     models = list(folder.glob("*_model.h5"))
    #     for mod in models:
    #         print(mod)
    #         f1 = main(folder/mod, write_yaml=True, plot=False, save_hist=True, eval_func=frame_wise_eval)
    #         results[folder].append(f1)
    #     #     stop = False
    #     #     import yaml
    #     #     while not stop:
    #     #         try:
    #     #             main(folder/mod, write_yaml=True, plot=True, save_hist=True)
    #     #         except:
    #     #             continue
    #     #         #check_perf
    #     #         with open(str(folder/mod).replace('model.h5', 'settings.yaml')) as file:
    #     #             target = yaml.load(file, Loader=yaml.FullLoader)
    #     #         with open(str(folder/mod).replace('model.h5', 'f1_settings.yaml')) as file:
    #     #             settings = yaml.load(file, Loader=yaml.FullLoader)
    #     #         try:
    #     #             print(settings['performance']['fbeta'], 'target:', target['performance']['fbeta'])
    #     #         except:
    #     #             print('no settings file')
    #     #             stop= True
    #     #             continue
    #     #         if settings['performance']['fbeta'] == target['performance']['fbeta']:
    #     #             stop = True

    # print(results)
    # for key, result in results.items():
    #     print(key)
    #     print(np.mean(results[key]))

    #### LIPID DROPLETS
    main_folder = Path("x:/Scientific_projects/deep_events_WS/data/original_data/training_data/")
    dirs = []


    # dirs.extend(list(main_folder.glob('20241102_0635*')))
    # dirs.extend(list(main_folder.glob('20241123_033*')))
    # dirs.extend(list(main_folder.glob('20241203_120*')))
    # dirs.extend(list(main_folder.glob('20241101_215*')))
    # dirs.extend(list(main_folder.glob('20241122_180*')))
    # dirs.extend(list(main_folder.glob('20241130_034*')))
    # dirs.extend(list(main_folder.glob('20241210_180*')))
    #new LD
    # dirs.extend(list(main_folder.glob('20250130_*')))
    dirs.extend(list(main_folder.glob('20250205_2131*')))
    dirs.extend(list(main_folder.glob('20250206_*')))

    for folder in dirs:
        models = list(folder.glob("*_model.h5"))
        for mod in models:
            print(mod)
            f1 = main(folder/mod, write_yaml=True, plot=True, save_hist=True, eval_func=whole_ev_eval)

