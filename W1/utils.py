import cv2
# import ffmpeg

# (
#     ffmpeg
#         .input('./data/AICity_data/train/S03/c010/vdo.avi')
#         .filter('fps', fps=1)
#         .output('test/%d.png',
#                 start_number=0)
#         .run(capture_stdout=True, capture_stderr=True)
#  )


colors = {
    'r': (0, 0, 255),
    'g': (0, 255, 0),
    'b': (255, 0, 0)
}


def draw_boxes(image, boxes, color='g', linewidth=2, det=False):
    rgb = colors[color]
    for box in boxes:
        image = cv2.rectangle(image, (int(box.xtl), int(box.ytl)), (int(box.xbr), int(box.ybr)), rgb, linewidth)
        if det:
            cv2.putText(image, str(box.confidence), (int(box.xtl), int(box.ytl) - 5), cv2.FONT_ITALIC, 0.6, rgb, linewidth)
    return image
