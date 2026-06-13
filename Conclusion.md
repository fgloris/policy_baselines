lowdim setting:
1. diffusion policy 在多步(100 steps) 略优于 flow matching
2. diffusion policy 在中等步(60 steps) 略劣于 flow matching
3. diffusion policy 在 1300 步开始过拟合，flow matching 在 900 步开始过拟合
4. (60 steps) 单纯将 diffusion epsilon 输出改成 x，沿用 MSE loss，会严重掉点: test mean score 0.7->0.5。用上完整 JiT 的 setting 掉点少一些但还是比较差。不要尝试 JiT 了。
5. 纯 MLP 作分类加分类内回归有搞头

待尝试 baseline:
1. consistency flow matching
2. sfp / action to action