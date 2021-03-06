import cv2
# import ffmpeg
import imageio
import os
import numpy as np

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
    'b': (255, 0, 0),
    'w': (255,255,255)
}
color_ids = {}


def draw_boxes(image, boxes, tracker=None, color='g', linewidth=2, det=False, boxIds=False, old=False):
    rgb = colors[color]
    for box in boxes:
        # print(box.id)
        if boxIds:
            if box.id in list(color_ids.keys()):
                pass
            else:
                color_ids[box.id]=np.random.uniform(0,256,size=3)
            if old:
                cv2.putText(image, str(box.id), (int(box.xtl), int(box.ytl) + 120), cv2.FONT_ITALIC, 0.6, color_ids[box.id], linewidth)
            else:
                cv2.putText(image, str(box.id), (int(box.xtl), int(box.ytl) + 20), cv2.FONT_ITALIC, 0.6, color_ids[box.id], linewidth)

            if tracker is not None:
                if box.id in tracker:
                    if len(tracker[box.id])>2:
                        image =cv2.polylines(image,[np.array(tracker[box.id])],False,color_ids[box.id],linewidth)

            # if len(kalman_predictions[box.id])>2:
            #     image =cv2.polylines(image,[np.array(kalman_predictions[box.id])],False,color_ids[box.id],linewidth)


            image = cv2.rectangle(image, (int(box.xtl), int(box.ytl)), (int(box.xbr), int(box.ybr)), color_ids[box.id], linewidth)
        else:
            image = cv2.rectangle(image, (int(box.xtl), int(box.ytl)), (int(box.xbr), int(box.ybr)), rgb, linewidth)

        if det:
            cv2.putText(image, str(box.confidence), (int(box.xtl), int(box.ytl) - 5), cv2.FONT_ITALIC, 0.6, rgb, linewidth)
       
    return image

def draw_boxes_old(image, boxes, tracker,  color='g', linewidth=2, det=False, boxIds=False, old=False, shifted=False):
    rgb = colors[color]
    for box in boxes:
        if not shifted:
            image = cv2.rectangle(image, (int(box.xtl), int(box.ytl)), (int(box.xbr), int(box.ybr)), colors['r'], linewidth)
        else:
            image = cv2.rectangle(image, (int(box.box_flow[0]), int(box.box_flow[1])), (int(box.box_flow[2]), int(box.box_flow[3])), colors['b'], linewidth)
    return image


def save_gif(source_path, results_path, fps=10):
    # Build GIF

    with imageio.get_writer(results_path, mode='I', fps=fps) as writer:
        for filename in sorted(os.listdir(source_path)):
            if filename[0] != ".":
                # print(source_path + filename)
                image = imageio.imread(source_path + filename)
                writer.append_data(image)


def plot_3D(results_path):
    maps=[]
    for i in x:
        map_alfa = []
        for j in y:
            z = uniform(0.0,1) 
            map_alfa.append(z)
        maps.append(map_alfa)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, np.array(maps),cmap='viridis', edgecolor='none')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Rho')
    ax.set_zlabel('mAP')
    # Add a color bar which maps values to colors.
    ax.view_init(45, 35)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(results_path+"grid_search.png")
    plt.show()