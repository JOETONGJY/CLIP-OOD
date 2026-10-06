# Δw 案例库（第三章机制证据）

> Δw = w_final − w_prior（epoch 100 终值, seed 1; 语义见实验记录）。

| 数据集 | 被上调 Top-8 (prompt, Δw, r, w_final) | 被下调 Top-8 | corr(Δw, r) |
|---|---|---|---|
| AWA2 | postage stamp illustration (+2.103, r=0.90)<br>stained glass panel (+1.973, r=0.91)<br>pastel drawing (+1.923, r=0.92)<br>paper cutout (+1.922, r=0.91)<br>laser-cut wood model (+1.900, r=0.91)<br>metal engraving (+1.885, r=0.88)<br>linocut (+1.883, r=0.90)<br>linocut print (+1.857, r=0.91) | photo collage (-0.779, r=0.92)<br>photo negative (-0.774, r=0.89)<br>aerial view photo (-0.755, r=0.90)<br>wireframe model (-0.754, r=0.89)<br>collage (-0.754, r=0.91)<br>sand sculpture (-0.746, r=0.89)<br>motion graphic (-0.744, r=0.86)<br>carved pumpkin (-0.740, r=0.87) | +0.488 |
| CUB | impressionist painting (+0.388, r=0.87)<br>painting (+0.385, r=0.88)<br>photo collage (+0.381, r=0.93)<br>watercolor (+0.376, r=0.90)<br>constellation (+0.375, r=0.76)<br>sketch (+0.371, r=0.88)<br>tapestry (+0.368, r=0.86)<br>doodle (+0.366, r=0.86) | woodcut (-0.302, r=0.87)<br>aerial view photo (-0.302, r=0.88)<br>wire sculpture (-0.299, r=0.85)<br>pottery design (-0.297, r=0.83)<br>cave wall painting (-0.295, r=0.86)<br>origami model (-0.295, r=0.83)<br>bronze statue (-0.293, r=0.83)<br>hyperrealistic drawing (-0.290, r=0.83) | +0.514 |