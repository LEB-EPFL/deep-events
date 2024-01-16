from pathlib import Path
import os

import tifffile
import tensorflow as tf
import numpy as np
from benedict import benedict

from deep_events.database.convenience import get_latest_folder, get_latest
from deep_events.train import adjust_tf_dimensions
from mitosplit_net import evaluation


FOLDER = Path("W:/deep_events/data/training_data")


def main(model_dir: Path = None, write_yaml: bool = True):
    if model_dir is None:
        training_folder = Path(get_latest_folder(FOLDER)[0])
        model_dir = Path(get_latest("model", training_folder))
        print(model_dir)
    else:
        training_folder = model_dir.parent
    model = tf.keras.models.load_model(model_dir)

    eval_images = adjust_tf_dimensions(tifffile.imread(training_folder / "eval_images_00.tif"))
    frames = min(eval_images.shape[0], 300)
    eval_images = eval_images[:frames]
    eval_mask = adjust_tf_dimensions(tifffile.imread(training_folder / "eval_gt_00.tif"))[:frames]

    pred_output_test = evaluation.predict(eval_images, model)
    labels = evaluation.label(pred_output_test)
    labels = np.expand_dims(labels, axis=-1)
    true_labels = evaluation.label(eval_mask, eval_mask.max() / 2)

    stats = evaluation.fissionStatsStack(true_labels, labels)
    # [TP, FP, FN, TP_px, FP_px, FN_px]
    precision = evaluation.get_precision(stats[0], stats[1])
    tpr = evaluation.get_tpr(stats[0], stats[2])
    f1 = round(evaluation.get_f1_score(precision, tpr)*100)/100
    precision = round(precision*100)/100
    tpr = round(tpr*100)/100
    summary = f"""
    {model_dir}
    precision {precision}
    tpr {tpr}
    f1 {f1}
    """
    print(summary)

    if write_yaml:
        settings = benedict(str(model_dir).replace("model.h5", "settings.yaml"))
        settings["precision"] = precision
        settings["tpr"] = tpr
        settings["f1"] = f1
        settings.to_yaml(filepath=str(model_dir).replace("model.h5", "settings.yaml"))


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
        axs[0].imshow(output[0, :, :, 0], vmax=1)
        axs[0].set_title("prediction")
        axs[1].imshow(eval_mask[frame, :, :, 0])
        axs[1].set_title("ground truth")
        axs[2].imshow(eval_images[frame, :, :, 0])
        plt.show()
        frame = frame + 1

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#     main(Path("W:/deep_events/data/original_data/training_data/20230626_1508_brightfield_cos7/20230626_1509_model.h5"))
    # main(Path("W:/deep_events/data/original_data/training_data/20230626_1509_fluorescence_zeiss_cos7/20230626_1509_model.h5"))
    # main(Path("Z:/_Lab members/Juan/Experiments/230222_MitoSplitNet_TrainingSet_U2OS_iSIM/training_data/20230611_0201_isim_cos7/20230611_0202_model.h5"))
    folder = Path("W:/deep_events/data/original_data/training_data/20231028_1108_brightfield_cos7_n3_f0.5")
    model = "20231028_1111_model.h5"
    visual_eval(folder, model)
    # visual_eval(Path("W:/deep_events/data/original_data/training_data/20230718_0123_brightfield_cos7"),
    #             "20230718_0128_model.h5")