from src.logger import logger
import matplotlib.pyplot as plt
import os


def visualize(train_losses, test_accs):
    plt.figure(figsize=(12, 6), dpi=300)

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, "b-o", linewidth=2, markersize=4, label="Training Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(test_accs, "r-s", linewidth=2, markersize=4, label="Test Accuracy")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    final_acc = test_accs[-1] if test_accs else 0
    plt.suptitle(f"Training Result (Final Acc: {final_acc:.2%})", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(logger.log_dir, "curve.png"), bbox_inches="tight")
    plt.close()
