import numpy as np
import matplotlib.image as mpimage
from mog import magnitude_of_gradients
from mog import show_image


def seam_carve(image, seam_mask):
    """
    Removes a seam from the image depending on the seam mask. Returns an image
     that has one column less than <image>

    :param image:
    :param seam_mask:
    :return: smaller image
    """
    shrunken = image[seam_mask].reshape(
        (image.shape[0], -1, image[..., None].shape[2]))
    return shrunken.squeeze()


def update_global_mask(global_mask, new_mask):
    """
    Updates the global_mask that contains all previous seams by adding the new path contained in new_mask.

    :param global_mask: The global mask (bool-Matrix) where the new path should be added
    :param new_mask: The mask (bool-Matrix) containing a new path
    :return: updated global Mask
    """
    reduced_idc = np.indices(global_mask.shape)[
        :, ~global_mask][:, new_mask.flat]
    seam_mask = np.ones_like(global_mask, dtype=bool)
    seam_mask[reduced_idc[0], reduced_idc[1]] = False
    return seam_mask


def calculate_accum_energy(energy):
    """
    Function computes the accumulated energies

    :param energy: ndarray (float)
    :return: ndarray (float)
    """
    accumE = np.array(energy)
    ...
    for i in range(accumE.shape[0]):
        for j in range(accumE.shape[1]):
            if(i != 0 and j != 0):
                minimum = np.min(accumE[i - 1][j - 1: j + 2])
                accumE[i][j] += minimum

            elif(i != 0):
                minimum = np.min(accumE[i - 1][j: j + 2])

                accumE[i][j] += minimum

    return accumE


def create_seam_mask(accumE):
    """
    Creates and returns boolean matrix containing zeros (False) where to remove the seam

    :param accumE: ndarray (float)
    :return: ndarray (bool)
    """
    Mask = np.ones(accumE.shape, dtype=bool)
    ...
    for row in reversed(range(0, accumE.shape[0])):
        if(row == accumE.shape[0] - 1):
            index = np.argmin(accumE[row])
            Mask[row][index] = 0
        elif(index != 0):
            index = np.argmin(accumE[row][index - 1: index + 2]) + index - 1
            Mask[row][index] = 0
        else:
            index = np.argmin(accumE[row][index: index + 2]) + index
            Mask[row][index] = 0

    return(Mask)

    # ------------------------------------------------------------------------------
    # Main Bereich
    # ------------------------------------------------------------------------------
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Initalisierung
    # --------------------------------------------------------------------------
    # lädt das Bild
    img = mpimage.imread('bilder/tower.jpg')  # 'bilder/bird.jpg')

    # erstellt eine globale Maske
    # In der Maske sollen alle Pfade gespeichert werden die herrausgeschnitten wurden
    # An Anfang ist noch nichts herrausgeschnitten, also ist die Maske komplett False
    global_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    # Parameter einstellen:
    number_of_seams_to_remove = 200

    # erstellet das neue Bild, welches verkleinert wird
    new_img = np.array(img, copy=True)

    # --------------------------------------------------------------------------
    # Der Algorithmus
    # --------------------------------------------------------------------------
    # Für jeden Seam, der entfernt werden soll:
    for idx in range(number_of_seams_to_remove):
        ...
        energy = magnitude_of_gradients(new_img)
        accumE = calculate_accum_energy(energy)
        seam_mask = create_seam_mask(accumE)
        new_img = seam_carve(new_img, seam_mask)
        global_mask = update_global_mask(global_mask, seam_mask)
        copy_img = np.array(img, copy=True)
        copy_img[global_mask, :] = [255, 0, 0]
        mpimage.imsave("./new/smaller" + str(idx) + ".png", new_img)
        mpimage.imsave("./copy/copy" + str(idx) + ".png", copy_img)
        print(idx, " image carved:", img.shape)
        print(idx, " image carved:", new_img.shape)

    mpimage.imsave("./resultSeamCarving/result.png", new_img)
