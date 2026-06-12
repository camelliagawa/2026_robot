# Robot link meshes — FANUC LR Mate 200iD/7L (= 14L 外形)

`base_link.stl` 〜 `link_6.stl` は ROS-Industrial の
[fanuc_lrmate200id_support](https://github.com/ros-industrial/fanuc)
パッケージのビジュアルメッシュ（`lrmate200id` / `lrmate200id7l`）を
mm 単位へ変換し、描画負荷低減のため三角形数を間引いたものです。

- `link_2.stl` / `link_4.stl`: ロングアーム（7L = 14L と共通外形）用メッシュ
- その他: 標準 200iD と共通のメッシュ
- 元データ: 約 22,000 三角形 → 間引き後: 約 8,600 三角形

## License

Software License Agreement (BSD License)
Copyright (c) 2012-2015, TU Delft Robotics Institute. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions of the BSD
3-Clause License are met. See the upstream repository for the full text:
https://github.com/ros-industrial/fanuc/blob/melodic-devel/LICENSE
