from src import *

nr_steps = 50000
n_test = 100000
bs = 256
lr = 0.0001
runs = 10
eval_points = 200
test_steps = 5000


optimizerlist = []

optimizerlist.append(SGD(nr_steps = nr_steps,
    bs=bs,
    lr=lr,
    n_test=n_test,
    eval_points  = eval_points,
    with_norm = False,
    runs = runs,
    plotlabel = f"SGD",
    plotcolor="C0",
    linestyle="dashdot"
    ))

optimizerlist.append(SGDM(nr_steps = nr_steps,
    bs=bs,
    lr=lr,
    n_test=n_test,
    eval_points  = eval_points,
    with_norm = False,
    runs = runs,
    plotlabel = f"SGD momentum",
    plotcolor = "C6",
    linestyle="solid"
    ))

optimizerlist.append(ADAM(nr_steps = nr_steps,
    bs=bs,
    lr=lr,
    n_test=n_test,
    eval_points  = eval_points,
    with_norm = False,
    runs = runs,
    plotlabel = f"ADAM",
    plotcolor = "C2"
    ))

optimizerlist.append(ADAMW(nr_steps = nr_steps,
    bs=bs,
    lr=lr,
    n_test=n_test,
    eval_points  = eval_points,
    with_norm = False,
    runs = runs,
    plotlabel = f"ADAMW",
    plotcolor = "C3"
    ))

optimizerlist.append(arithm_average_adam_from_start(nr_steps = nr_steps,
    bs=bs,
    lr=lr,
    n_test=n_test,
    eval_points  = eval_points,
    with_norm = False,
    runs = runs,
    plotlabel = f"ADAM with Ruppert-Polyak average",
    plotcolor = "C5",
    linestyle = "dotted"
    ))

optimizerlist.append(geom_average_adam(nr_steps = nr_steps,
    bs=bs,
    lr=lr,
    n_test=n_test,
    eval_points  = eval_points,
    with_norm = False,
    runs = runs,
    gamma = 0.999,
    plotlabel = f"ADAM with EMA, delta = {0.999}",
    plotcolor = "C1"
    ))

optimizerlist.append(PADAM3(nr_steps = nr_steps,
    bs=bs,
    lr=lr,
    n_test=n_test,
    eval_points  = eval_points,
    with_norm = False,
    runs = runs,
    test_steps = test_steps,
    linewidth = 2,
    plotlabel = f"PADAM3",
    plotcolor = "C9"
    ))

optimizerlist.append(PADAM10(nr_steps = nr_steps,
    bs=bs,
    lr=lr,
    n_test=n_test,
    eval_points  = eval_points,
    with_norm = False,
    runs = runs,
    test_steps = test_steps,
    linewidth = 2,
    plotlabel = f"PADAM10",
    plotcolor = (177/255, 13/255,123/255),
    linestyle = "dashed"
    ))




model = configured_model("supervised")


result=test_optimizers(model = model, optimizer_list=optimizerlist, plot=True, savedata=True, savefig=True, save_logs=True, heading = "Supervised model approximating Gaussian density")