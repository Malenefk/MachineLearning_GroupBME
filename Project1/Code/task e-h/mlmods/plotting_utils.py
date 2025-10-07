import matplotlib.pyplot as plt

def plot_convergence(
    ax,
    conv_dict,
    title,
    xlabel,
    ylabel="Train MSE",
    fontsize_title=16,
    fontsize_label=14,
    fontsize_legend=12
):
    """
    Plots how the training error changes for each method on one axis.
    conv_dict is a dictionary where the key is the method name and the value is a list of MSE values over the iterations. 
    Adjustable font sizes for title, labels, and legend.
    """
    for name, series in conv_dict.items():
        if name == "Momentum":
            ax.plot(series, label=name, linewidth=1.5, linestyle="--")
        else:
            ax.plot(series, label=name, linewidth=1.5)
    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlabel(xlabel, fontsize=fontsize_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label)
    ax.tick_params(axis="both", labelsize=fontsize_label - 2)
    ax.grid(True)
    ax.legend(ncol=2, fontsize=fontsize_legend)
    

def plot_final_coeffs_with_cf(
    ax,
    family_dict,
    theta_cf,
    title,
    xlabel="Coefficient index",
    ylabel="Coefficient value",
    cf_label="CF",
    fontsize_title=16,
    fontsize_label=14,
    fontsize_legend=12
):
    """
    One axis: plot final coefficient vectors for a family, plus a black CF 
    reference.
    Adjustable font sizes for title, labels, and legend.
    """
    for name, (th, _) in family_dict.items():
        ax.plot(th.ravel(), linewidth=1.2, alpha=0.9, label=name)
    ax.plot(theta_cf.ravel(), 'k--', linewidth=1.5, label=cf_label, zorder=10)
    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlabel(xlabel, fontsize=fontsize_label)
    ax.set_ylabel(ylabel, fontsize=fontsize_label)
    ax.tick_params(axis="both", labelsize=fontsize_label - 2)
    ax.grid(True)
    ax.legend(ncol=2, fontsize=fontsize_legend)
