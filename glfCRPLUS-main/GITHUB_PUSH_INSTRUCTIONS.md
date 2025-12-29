# Instructions to Push Changes to GitHub

Since you are running this on Kaggle, you need to push the code changes I made to your GitHub repository.

## Step 1: Open Terminal (in this directory)
navigate to: `c:\Users\deepa\Desktop\Final Year Project\glfCR-modifed\glfCRPLUS-main`

## Step 2: Add and Commit Changes
Run the following commands:

```powershell
# Add all new files (including the enhancements folder)
git add .

# Commit the changes
git commit -m "Implemented Cross-Modal Attention and FFT Loss for Novelty"
```

## Step 3: Push to GitHub
```powershell
# Push to your main branch (or master)
git push origin main
```

*(Note: If your branch is named `master`, replace `main` with `master`)*

## Step 4: Verify on Kaggle
1. Open your Kaggle Notebook.
2. In the first cell where you clone the repo, make sure to pull the latest changes if you re-run the cell:
   ```bash
   !git clone https://github.com/your-username/your-repo-name.git
   # OR if already cloned
   %cd your-repo-name
   !git pull
   ```
3. Run the training script with the new flags: `--use_cross_attn` and `--use_fft_loss`.
