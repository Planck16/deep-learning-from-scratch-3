import math
pil_available = True
try:
    from PIL import Image
except:
    pil_available = False
import numpy as np
from dezero import cuda


