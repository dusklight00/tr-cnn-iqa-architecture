def rgb_to_grayscale(tensor):
    return 0.2989 * tensor[:, 0, :, :] + 0.5870 * tensor[:, 1, :, :] + 0.1140 * tensor[:, 2, :, :]
