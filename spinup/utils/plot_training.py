import numpy as np
import matplotlib.pyplot as plt

def plot_file(file_names, optimal_rews, title="", save_name="plot.png"):
    all_rews = []
    for fname, opt_rew in zip(file_names, optimal_rews):
        line_num, rews = 0, []
        with open(fname) as f:
            for line in f:
                print(line)
                print()
                line_num += 1
                if line_num >= 2 and len(line.split()) > 1:
                    rews.append(opt_rew - float(line.split()[1]))
        all_rews.append(rews)
    for rews in all_rews:
        print(rews)
        plt.plot(np.array(rews))
    plt.xlabel("Iterations")
    plt.ylabel("Regret")
    plt.title(title)
    plt.ylim([0, 10])
    plt.savefig(save_name)
    # plt.show()


if __name__ == "__main__":
    # file_names = ["data/210131/sac-upn-pyt-bench_pointirl-v0_alp0-1/sac-upn-pyt-bench_pointirl-v0_alp0-1_s0/progress.txt"]
    # optimal_rews = [-1.12]
    # file_names = ["data/210131/sac-upn-pyt-bench_pointirl-v01_alp0-1/sac-upn-pyt-bench_pointirl-v01_alp0-1_s0/progress.txt"]
    file_names = ["data/sac-upn-pyt-bench_pointirl-v1_alp0-1/sac-upn-pyt-bench_pointirl-v1_alp0-1_s0/progress.txt"]
    optimal_rews = [10]
    title = "GT Training"
    save_name = "point-v1.png"

    plot_file(file_names, optimal_rews, title, save_name)