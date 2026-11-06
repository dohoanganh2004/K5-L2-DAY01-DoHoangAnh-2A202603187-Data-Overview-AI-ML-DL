# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh

**File:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`, ảnh `sample_id = traffic`, dòng `rank = 1`.

1. **Record hạng 1:** `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về cả bức ảnh, không phải một vật thể riêng. Record chỉ có lớp, hạng và score, không có tọa độ hộp hay đa giác nào chỉ ra vị trí của "cab" trong ảnh.

3. Class list đi kèm taxonomy/checkpoint, model không tự nghĩ ra. Checkpoint này chỉ xếp hạng trong các lớp của ImageNet-1K, nên kết quả luôn là một lớp có sẵn trong danh sách, kể cả khi danh sách không mô tả đúng cảnh trong ảnh.

4. Cùng một mã hoặc tên có thể mang nghĩa khác nhau giữa các taxonomy. Khi ghi đủ cả ba trường giúp đọc đúng nhãn, tra lại được danh sách gốc và tránh nhầm khi gộp dữ liệu từ nhiều nguồn.

5. Guideline cần nói rõ tiêu chí chọn một lớp

6. `score` chỉ là mức tin của model vào kết quả của chính nó, không phải kết quả kiểm tra theo guideline. Model có thể  tin vào một kết quả không chính xác. Ground truth phải do người gán theo quy tắc, nếu lấy score làm ground truth thì là dùng chính dự đoán của model để tự chấm điểm model.

## 2. Phát hiện vật thể

**File** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`, ảnh `sample_id = kitchen`.

1. **Record:** `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Góc trên-trái của box ở `(385.33, 69.24)`, cách mép trái khoảng 385 pixel và cách mép trên khoảng 69 pixel. Góc dưới-phải ở `(498.92, 348.92)`. Box rộng khoảng 114 pixel, cao khoảng 280 pixel và bao quanh người đứng quay lưng ở phía sau bếp.

3. Hai ngưỡng đã chọn:
   - **`0.35`:** 11 prediction (2 `person`, 5 `bowl`, 2 `oven`, 2 `cup`).
   - **`0.60`:** 6 prediction (2 `person`, 2 `bowl`, 2 `oven`).

4. Khi hạ ngưỡng, reviewer phải xem nhiều hộp hơn, loại hộp sai lớp, hộp trùng và hộp lệch, đồng thời kiểm tra vật có score thấp nhưng thực sự tồn tại. Ngưỡng chỉ lọc prediction, không phải quy tắc để bỏ vật thể khi tạo ground truth.

5. **Quy tắc box:** Mỗi vật thuộc taxonomy có một hộp riêng, ôm sát phần nhìn thấy của vật, không chừa thêm nền và không ước lượng phần bị che hoặc nằm ngoài ảnh.

6. **Vật bị cắt mép:** `person` score 0.610931 tại `[0.12, 263.18, 61.55, 310.78]` chỉ là một cánh tay nằm sát mép trái ảnh . Vì vật bị cắt mép lặp lại ở nhiều ảnh và xử lý được theo một cách thống nhất: vẫn gán `person` và vẽ hộp theo phần nhìn thấy. Chỉ escalation khi phần nhìn thấy quá ít để biết chắc đó là vật gì.

## 3. Phân đoạn theo từng đối tượng

**File:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`, ảnh `sample_id = kitchen`.

1. **Record:** `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]

2. Box chỉ cho biết vị trí và kích thước bao ngoài, nên vẫn chứa cả phần nền. Polygon bám theo đường viền thật của vật, nên thấy được dáng đầu, tay và chân của người, và phân biệt được pixel thuộc người với pixel nền bên trong hộp.

3. `instance_id` dùng để phân biệt từng đối tượng trong output của bài lab. Nó không phải class ID vì `kitchen-001` và `kitchen-009` có `instance_id` khác nhau nhưng cùng `class_id`=0 (`person`). Nó không phải tracking ID vì chỉ là số thứ tự trong output của một ảnh, không theo dõi cùng một vật qua nhiều khung hình.

4. **Quy tắc vẽ biên mask:** Vẽ polygon bám sát biên phần nhìn thấy của vật, mỗi instance một mask riêng; không tính vật khác đang đặt lên hoặc che phía trước vào mask.


## 4. Vòng đời và kiểm tra chất lượng

Chuỗi vòng đời production: `ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho toàn ảnh | `traffic`: top-1 là `cab`  dù ảnh có nhiều xe buýt, ô tô và người; không có một chủ thể rõ ràng | Chọn một lớp theo tiêu chí trong guideline; escalation nếu không có lớp phù hợp | Lớp có đúng taxonomy và đúng tiêu chí chọn chủ thể không |
| Phát hiện vật thể | Một lớp và một hộp cho mỗi vật thể | `kitchen`: xoong chảo treo không có box vì COCO-80 không có lớp này | Vẽ hộp ôm sát phần nhìn thấy cho từng vật | Hộp có sát không, có sai lớp, trùng hoặc bỏ sót vật không |
| Instance segmentation | Một lớp và một đa giác cho mỗi instance | `kitchen-004`: lá treo được gán `potted plant` (0.632235) dù không thấy chậu; `kitchen-005`: mask `dining table` phủ cả vật trên bàn | Vẽ biên sát từng instance, tách vật chạm nhau, không gộp vật đặt lên trên | Biên mask có sát không, instance có bị gộp hay lẫn vật khác không, lớp có đúng không |

## 5. An toàn dữ liệu

1. Trong bài lab này, mình chỉ dùng ba ảnh COCO công khai có sẵn trong notebook, không tải lên ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, mình sẽ dừng lại, không dùng tiếp và báo cho GV/Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
