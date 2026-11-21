import armModelUrl from './assets/upper-limb.glb?url';
import arm26 from './data/arm26.json';

const legModels = import.meta.glob('../lower-limb/assets/lower-limb.glb',
  { eager: true, import: 'default', query: '?url' });
const legData = import.meta.glob('../lower-limb/data/leg-mechanics.json',
  { eager: true, import: 'default' });
const modelRevision = 'd9b05d470b1a481c222372c85b75772faf8f7792';

export const arm = {
  key: 'arm',
  title: 'Upper limb',
  joint: 'Elbow',
  animation: 'Elbow flexion',
  pivot: 'ElbowPivot',
  maxAngle: 125,
  startAngle: 65,
  detailSize: [0.155, 0.182, 0.165],
  motion: 'curl',
  modelUrl: armModelUrl,
  embeddedModelId: 'embedded-model',
  filename: 'upper-limb.glb',
  loadSegment: 'Hand',
  leverM: 0.30,
  loadCenter: 'HandLoadCenter',
  muscle: 'BIClong',
  muscleLabel: 'Biceps',
  mechanics: arm26,
  mechanicsNote: 'Torque uses the external weight and a hypothetical 30 cm lever. Biceps values refer to the long head.',
  nerves: {
    'Median nerve': {
      description: 'Runs in front of the elbow, usually between the heads of pronator teres.',
      url: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC4579740/',
    },
    'Ulnar nerve': {
      description: 'Passes behind the medial epicondyle through the cubital tunnel. Elbow flexion changes the tunnel’s shape.',
      url: 'https://pubmed.ncbi.nlm.nih.gov/9563378/',
    },
    'Radial nerve': {
      description: 'Reaches the lateral elbow, where it divides into superficial and deep branches.',
      url: 'https://pubmed.ncbi.nlm.nih.gov/18795220/',
    },
    'Musculocutaneous nerve': {
      description: 'Supplies the biceps and brachialis in the front of the upper arm.',
      url: 'https://pubmed.ncbi.nlm.nih.gov/7594300/',
    },
  },
};

export const legs = {
  key: 'legs',
  title: 'Lower limb',
  joint: 'Knee',
  animation: 'Knee flexion',
  pivot: 'KneePivot',
  maxAngle: 120,
  startAngle: 45,
  detailSize: [0.225, 0.245, 0.215],
  motion: 'flexion',
  modelUrl: legModels['../lower-limb/assets/lower-limb.glb'],
  embeddedModelId: 'embedded-leg-model',
  filename: 'lower-limb-right.glb',
  loadSegment: 'Ankle',
  leverM: 0.40,
  loadCenter: 'FootLoadCenter',
  muscle: 'recfem_r',
  muscleLabel: 'Rectus femoris',
  mechanics: legData['../lower-limb/data/leg-mechanics.json'],
  mechanicsNote: 'Torque uses the external weight and a hypothetical 40 cm lever. Negative muscle moment arms indicate knee extension.',
  modelReference: `https://github.com/opensim-org/opensim-models/blob/${modelRevision}/Models/Rajagopal/Rajagopal2016.osim`,
  atlasArchive: 'https://caskanatomy.info/open3dmodelfiles/lower-limb/lower-limb-glb.zip',
  nerves: {
    'Sciatic nerve': {
      description: 'Descends behind the thigh and divides into the tibial and common fibular nerves near the knee. The division level varies.',
      url: 'https://pubmed.ncbi.nlm.nih.gov/34052135/',
    },
    'Tibial nerve': {
      description: 'Continues through the back of the knee into the posterior lower leg.',
      url: 'https://pubmed.ncbi.nlm.nih.gov/19934998/',
    },
    'Common fibular nerve': {
      description: 'Courses along the outer knee and wraps near the neck of the fibula.',
      url: 'https://pubmed.ncbi.nlm.nih.gov/38510322/',
    },
    'Femoral nerve': {
      description: 'Enters the front of the thigh and gives off muscular and sensory branches.',
      url: 'https://link.springer.com/article/10.1007/s12565-026-00933-x',
    },
  },
};
