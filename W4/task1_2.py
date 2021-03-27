import pyflow

sys.path.append("W1")
from flow_evaluation.py import compute_pepn,compute_msen

im1 = np.array(Image.open('examples/car1.jpg'))
im2 = np.array(Image.open('examples/car2.jpg'))
im1 = im1.astype(float) / 255.
im2 = im2.astype(float) / 255.

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))



