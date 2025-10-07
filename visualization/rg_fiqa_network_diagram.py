"""
RG-FIQA network structure visualization.
Modified from visualization/RG-FIQA_net.py.
Draw the network defined in rg_fiqa/models/rg_fiqa.py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

# Visualization parameters
NumDots = 4
NumConvMax = 8
NumFcMax = 20

# Professional color scheme
White = 1.
Light_Blue = np.array([0.8, 0.9, 1.0])      # 浅蓝色 - 输入层
Light_Green = np.array([0.8, 1.0, 0.8])     # 浅绿色 - 卷积层
Light_Orange = np.array([1.0, 0.9, 0.7])    # 浅橙色 - 池化层
Light_Purple = np.array([0.9, 0.8, 1.0])    # 浅紫色 - 全连接层
Medium_Gray = np.array([0.6, 0.6, 0.6])     # 中灰色 - 连接线
Dark_Gray = np.array([0.3, 0.3, 0.3])       # 深灰色 - 边框
Black = np.array([0., 0., 0.])


def add_layer(patches, colors, size=(24, 24), num=5,
              top_left=[0, 0],
              loc_diff=[3, -3],
              layer_type='conv'):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    
    # Choose color by layer type
    if layer_type == 'input':
        base_color = Light_Blue
    elif layer_type == 'conv':
        base_color = Light_Green
    elif layer_type == 'pool':
        base_color = Light_Orange
    elif layer_type == 'fc':
        base_color = Light_Purple
    else:
        base_color = Light_Green
    
    for ind in range(num):
        patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
        # Slight color variation for depth effect
        color_variation = 0.9 + 0.1 * (ind % 2)
        colors.append(base_color * color_variation)


def add_layer_with_omission(patches, colors, size=(24, 24),
                            num=5, num_max=8,
                            num_dots=4,
                            top_left=[0, 0],
                            loc_diff=[3, -3],
                            layer_type='conv'):
    # add a rectangle
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    this_num = min(num, num_max)
    start_omit = (this_num - num_dots) // 2
    end_omit = this_num - start_omit
    start_omit -= 1
    
    # Choose color by layer type
    if layer_type == 'input':
        base_color = Light_Blue
    elif layer_type == 'conv':
        base_color = Light_Green
    elif layer_type == 'pool':
        base_color = Light_Orange
    elif layer_type == 'fc':
        base_color = Light_Purple
    else:
        base_color = Light_Green
    
    for ind in range(this_num):
        if (num > num_max) and (start_omit < ind < end_omit):
            omit = True
        else:
            omit = False

        if omit:
            patches.append(
                Circle(loc_start + ind * loc_diff + np.array(size) / 2, 0.8))
        else:
            patches.append(Rectangle(loc_start + ind * loc_diff,
                                     size[1], size[0]))

        if omit:
            colors.append(Dark_Gray)
        else:
            # Slight color variation for depth effect
            color_variation = 0.9 + 0.1 * (ind % 2)
            colors.append(base_color * color_variation)


def add_mapping(patches, colors, start_ratio, end_ratio, patch_size, ind_bgn,
                top_left_list, loc_diff_list, num_show_list, size_list, operation_type='conv'):

    start_loc = top_left_list[ind_bgn] \
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
        + np.array([start_ratio[0] * (size_list[ind_bgn][1] - patch_size[1]),
                    - start_ratio[1] * (size_list[ind_bgn][0] - patch_size[0])]
                   )

    end_loc = top_left_list[ind_bgn + 1] \
        + (num_show_list[ind_bgn + 1] - 1) * np.array(
            loc_diff_list[ind_bgn + 1]) \
        + np.array([end_ratio[0] * size_list[ind_bgn + 1][1],
                    - end_ratio[1] * size_list[ind_bgn + 1][0]])

    # Kernel color
    if operation_type == 'conv':
        kernel_color = Light_Orange * 0.8
    elif operation_type == 'pool':
        kernel_color = Light_Purple * 0.8
    else:
        kernel_color = Medium_Gray

    patches.append(Rectangle(start_loc, patch_size[1], -patch_size[0]))
    colors.append(kernel_color)
    
    # Improved connection line style
    line_color = Medium_Gray
    line_width = 1.5
    
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1], end_loc[1]], linewidth=line_width))
    colors.append(line_color)
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
                          [start_loc[1], end_loc[1]], linewidth=line_width))
    colors.append(line_color)
    patches.append(Line2D([start_loc[0], end_loc[0]],
                          [start_loc[1] - patch_size[0], end_loc[1]], linewidth=line_width))
    colors.append(line_color)
    patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
                          [start_loc[1] - patch_size[0], end_loc[1]], linewidth=line_width))
    colors.append(line_color)


def label(xy, text, xy_off=[0, 4], fontsize=9, fontweight='normal', color='black'):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='Arial', size=fontsize, fontweight=fontweight, 
             color=color, ha='center', va='center')


def create_rg_fiqa_diagram():
    fc_unit_size = 3
    layer_width = 55
    flag_omit = True

    patches = []
    colors = []

    # Style settings
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    fig, ax = plt.subplots(figsize=(16, 7))

    ############################
    # RG-FIQA conv layers
    # Structure: Input(112×112) → Conv1(112×112) → Conv2(56×56) → Conv3(28×28) → Conv4(14×14) → Conv5(7×7) → GAP(1×1)
    
    # Layer sizes (height, width)
    size_list = [(40, 40), (36, 36), (30, 30), (24, 24), (18, 18), (12, 12), (6, 6)]
    # Channels
    num_list = [3, 32, 64, 128, 256, 512, 512]  # 最后一个是GAP后的特征
    # Resolutions for labels
    resolution_list = [112, 112, 56, 28, 14, 7, 1]
    # Layer types
    layer_types = ['input', 'conv', 'conv', 'conv', 'conv', 'conv', 'pool']
    
    x_diff_list = [0] + [layer_width] * (len(size_list) - 1)
    text_list = ['Input'] + [f'Conv{i}' for i in range(1, 6)] + ['GAP']
    loc_diff_list = [[3, -3]] * len(size_list)

    num_show_list = list(map(min, num_list, [NumConvMax] * len(num_list)))
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

    # Draw conv layers
    for ind in range(len(size_list)-1, -1, -1):
        if flag_omit and num_list[ind] > 1:
            add_layer_with_omission(patches, colors, size=size_list[ind],
                                    num=num_list[ind],
                                    num_max=NumConvMax,
                                    num_dots=NumDots,
                                    top_left=top_left_list[ind],
                                    loc_diff=loc_diff_list[ind],
                                    layer_type=layer_types[ind])
        else:
            add_layer(patches, colors, size=size_list[ind],
                      num=num_show_list[ind],
                      top_left=top_left_list[ind], 
                      loc_diff=loc_diff_list[ind],
                      layer_type=layer_types[ind])
        
        # Add layer labels
        if ind == 0:
            label_text = f'{text_list[ind]}\n{num_list[ind]}@{resolution_list[ind]}×{resolution_list[ind]}'
        else:
            label_text = f'{text_list[ind]}\n{num_list[ind]}@{resolution_list[ind]}×{resolution_list[ind]}'
        
        # Adjust label position to avoid overlap
        label(top_left_list[ind], label_text, xy_off=[0, 15], fontsize=10, fontweight='bold')

    ############################
    # Connections between conv layers (kernels and downsampling)
    start_ratio_list = [[0.4, 0.5]] * (len(size_list) - 2)  # 除了最后一个GAP
    end_ratio_list = [[0.4, 0.5]] * (len(size_list) - 2)
    patch_size_list = [(3, 3)] * (len(size_list) - 2)  # 都是3x3卷积核
    operation_types = ['conv'] * (len(size_list) - 2)
    text_list_conv = [f'Conv 3×3\nStride {stride}' for stride in [1, 2, 2, 2, 2]]
    
    # Add GAP connection
    start_ratio_list.append([0.4, 0.5])
    end_ratio_list.append([0.4, 0.5])
    patch_size_list.append((2, 2))
    operation_types.append('pool')
    text_list_conv.append('Global\nAvgPool')

    for ind in range(len(patch_size_list)):
        if ind < len(size_list) - 1:
            add_mapping(
                patches, colors, start_ratio_list[ind], end_ratio_list[ind],
                patch_size_list[ind], ind,
                top_left_list, loc_diff_list, num_show_list, size_list,
                operation_type=operation_types[ind])
            
            # Operation labels
            if ind < len(text_list_conv):
                label(top_left_list[ind], text_list_conv[ind], 
                     xy_off=[35, -65], fontsize=9, fontweight='normal')

    ############################
    # Fully connected layer
    size_list_fc = [(fc_unit_size*2, fc_unit_size*4)]
    num_list_fc = [1]
    num_show_list_fc = [1]
    
    x_diff_list_fc = [sum(x_diff_list) + layer_width + 10]
    top_left_list_fc = np.c_[np.cumsum(x_diff_list_fc), [0]]
    loc_diff_list_fc = [[fc_unit_size, -fc_unit_size]]

    # Draw FC layer
    add_layer(patches, colors, size=size_list_fc[0],
              num=num_show_list_fc[0],
              top_left=top_left_list_fc[0],
              loc_diff=loc_diff_list_fc[0],
              layer_type='fc')
    
    # Output layer label
    label(top_left_list_fc[0], 'Output\n1 (Quality Score)', 
          xy_off=[0, 15], fontsize=10, fontweight='bold')

    # FC connection line
    start_loc = top_left_list[-1] + (num_show_list[-1] - 1) * np.array(loc_diff_list[-1])
    end_loc = top_left_list_fc[0]
    line_color = Medium_Gray
    patches.append(Line2D([start_loc[0] + size_list[-1][1], end_loc[0]],
                          [start_loc[1] - size_list[-1][0]/2, end_loc[1] - fc_unit_size*2],
                          linewidth=2.0, color=line_color))
    colors.append(line_color)
    
    # FC operation label
    label(top_left_list_fc[0], 'Linear + Sigmoid', 
          xy_off=[-20, -35], fontsize=9, fontweight='normal')

    ############################
    # Render
    for patch, color in zip(patches, colors):
        if isinstance(patch, Line2D):
            # Line style
            if hasattr(patch, 'get_color') and patch.get_color() is not None:
                # 如果已经设置了颜色，保持原有颜色
                pass
            else:
                patch.set_color(color if isinstance(color, np.ndarray) else color * np.ones(3))
            patch.set_alpha(0.8)
            ax.add_line(patch)
        else:
            # Shape style
            if isinstance(color, np.ndarray):
                patch.set_facecolor(color)
            else:
                patch.set_color(color * np.ones(3))
            patch.set_edgecolor(Dark_Gray)
            patch.set_linewidth(0.8)
            patch.set_alpha(0.9)
            ax.add_patch(patch)

    # Layout
    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    
    # White background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Figure size
    fig.set_size_inches(16, 7)
    
    return fig, ax


if __name__ == '__main__':
    fig, ax = create_rg_fiqa_diagram()
    
    # Save figure
    fig_dir = './visualization'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    fig_path = os.path.join(fig_dir, 'rg_fiqa_network_structure.png')
    fig.savefig(fig_path, bbox_inches='tight', pad_inches=0.3, 
                dpi=300, facecolor='white', edgecolor='none',
                format='png', transparent=False)
    
    print(f"RG-FIQA network diagram saved to: {fig_path}")
    plt.show()
