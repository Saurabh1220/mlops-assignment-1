from sklearn.kernel_ridge import KernelRidge
from misc import load_boston_df, split_xy, train_and_eval

def main():
    df = load_boston_df()
    X_tr, X_te, y_tr, y_te = split_xy(df, target_col="MEDV", test_size=0.2, seed=42)
    model = KernelRidge(kernel="rbf", alpha=1.0, gamma=None)
    mse = train_and_eval(model, X_tr, y_tr, X_te, y_te)
    print(f"[KernelRidge] Test MSE: {mse:.4f}")

if __name__ == "__main__":
    main()
