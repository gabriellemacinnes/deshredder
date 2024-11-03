import os
import time
import json
import argparse
import random
import matplotlib.pyplot as plt

from docrec.metrics import accuracy
from docrec.strips import Strips
from docrec.compatibility.proposed import Proposed
from docrec.pipeline import Pipeline
from docrec.solverconcorde import SolverConcorde


NUM_CLASSES = 2

def reconstruct(doc, output_dir, thresh='sauvola', model_id='cdip_0.2_1000_32x64_128_fire3_1.0_0.1', 
                  input_size=[3000, 64], vshift=10, feat_dim=128, feat_layer='fire3', activation='sigmoid'):

    # algorithm definition
    weights_path_left = json.load(open('traindata/{}/info.json'.format(model_id), 'r'))['best_model_left']
    weights_path_right = json.load(open('traindata/{}/info.json'.format(model_id), 'r'))['best_model_right']
    sample_height = json.load(open('traindata/{}/info.json'.format(model_id), 'r'))['sample_height']
    algorithm = Proposed(
        weights_path_left, weights_path_right, vshift,
        input_size, feat_dim=feat_dim, feat_layer=feat_layer,
        activation=activation, sample_height=sample_height,
        thresh_method=thresh
    )

    # pipeline: compatibility algorithm + solver
    solver = SolverConcorde(maximize=False, max_precision=2)
    pipeline = Pipeline(algorithm, solver)

    # load strips and shuffle the strips
    print('1) Load strips')

    strips = Strips(path=doc, filter_blanks=True)
    strips.shuffle()
    init_permutation = strips.permutation()
    print('Shuffled order: ' + str(init_permutation))

    print('2) Results')
    solution, compatibilities, displacements = pipeline.run(strips)
    displacements = [displacements[prev][curr] for prev, curr in zip(solution[: -1], solution[1 :])]
    corrected = [init_permutation[idx] for idx in solution]
    print('Solution: ' + str(solution))
    print('Correct order: ' + str(corrected))
    print('Accuracy={:.2f}%'.format(100 * accuracy(solution, init_permutation)))
    reconstruction = strips.image(order=solution, displacements=displacements)
    
    plt.imshow(reconstruction, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'reconstruction.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

# reconstruct('imgs/D016', 'Deshredder/output')