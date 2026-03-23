import nibabel as nib
import numpy as np
import argparse
import os
from pathlib import Path


def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata(), img.affine


def save_nifti(data, affine, path):
    nib.save(nib.Nifti1Image(data, affine), path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub-dir', default='./output')
    parser.add_argument('--pre-dir', default='data/processed_pre')
    parser.add_argument('--out-dir', default='./output_post')
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()

    sub_dir = Path(args.sub_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 统计
    found_count = 0
    missing_count = 0

    for sub_file in sorted(sub_dir.glob('*_sub.nii.gz')):
        # 提取 ID 和侧别: DUKE_021_L_sub.nii.gz -> DUKE_021_L
        # stem 是 DUKE_021_L_sub.nii，去掉 _sub.nii 得到 DUKE_021_L
        base_name = sub_file.name.replace('_sub.nii.gz', '')
        pre_file = Path(args.pre_dir) / f"{base_name}_pre_aligned.nii.gz"

        if not pre_file.exists():
            print(f"Warning: {pre_file} not found, skipping")
            missing_count += 1
            continue

        # 加载数据
        pre_data, affine = load_nifti(pre_file)
        sub_data, _ = load_nifti(sub_file)

        # 计算 post = pre + alpha * sub
        post_data = pre_data + args.alpha * sub_data

        # 保存
        out_file = out_dir / f"{base_name}_post.nii.gz"
        save_nifti(post_data, affine, out_file)
        print(f"Saved: {out_file}")
        found_count += 1

    print(f"\nDone! Processed {found_count} files, {missing_count} missing pre images.")


if __name__ == '__main__':
    main()
