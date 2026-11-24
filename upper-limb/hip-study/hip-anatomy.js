import * as THREE from 'three';

const MEDIUS = 'Gluteus medius muscle';
const MINIMUS = 'Gluteus minimus muscle';
const MAXIMUS = 'Gluteus maximus muscle';
const HEAD = 'Art cart of femur head';
const SOCKET = 'Articular cartilage of hip bone acetabulum';
const LABRUM = 'Acetabular labrum';
const BAND = 'Iliotibial tract';
const CAPSULE = 'Hip joint capsule';

// All anchors are snapped to vertices on the named source structure. XYZ is
// only a search seed in the source's neutral, right-sided coordinate system.
const anchor = (label, text, xyz, side, compact = text) => ({label, text, compact, xyz, side});
const gluteal = anchor(MEDIUS,'Gluteus medius',[-.139,.943,-.042],-1,'Glut. medius');
const trochanter = anchor('Femur','Greater trochanter',[-.146,.845,-.022],-1,'Trochanter');
const head = anchor(HEAD,'Femoral head',[-.09,.866,.011],1,'Head');
export const HIP_PROFILES = {
  surface: {
    direction:[-1.18,.12,-.84], target:[.003,-.004,-.021], halfHeight:.171, halfWidth:.107,
    anchors:[anchor(MAXIMUS,'Gluteus maximus',[-.11,.883,-.099],-1,'Glut. maximus'),
      anchor(BAND,'Iliotibial band',[-.155,.79,-.006],1,'IT band'),
      anchor('Tensor fasciae latae','Tensor fasciae latae',[-.142,.913,.023],1,'TFL')],
  },
  deep: {
    direction:[-.94,.10,-1.12], target:[.003,.013,-.027], halfHeight:.152, halfWidth:.093,
    anchors:[anchor(MINIMUS,'Gluteus minimus',[-.127,.933,-.025],1,'Glut. minimus'),
      anchor('Trochanteric bursa of gluteus maximus','Trochanteric bursa',[-.14283,.84064,-.02717],1,'Bursa'),
      anchor('Piriformis muscle','Piriformis',[-.085,.871,-.054],-1)],
  },
  joint: {
    direction:[-1.12,-.24,.67], target:[-.004,-.013,-.012], halfHeight:.082, halfWidth:.086,
    anchors:[anchor(LABRUM,'Acetabular labrum',[-.099,.877,.010],1,'Labrum'),
      {...head, side:-1}, anchor(SOCKET,'Acetabular cartilage',[-.071,.870,.011],1,'Socket cartilage')],
  },
  overview: {
    direction:[-1.24,.18,1],target:[.006,.005,-.013],halfHeight:.181,halfWidth:.113,
    anchors:[gluteal,trochanter,head],
  },
};

const sourceLayer = {
  bones:'bone', muscles:'muscle', tendons:'tendons', nerves:'nerves', fascia:'fascia',
  arteries:'vessels', veins:'vessels', vessels:'vessels', ligaments:'ligaments', cartilage:'cartilage', bursae:'bursae',
};
const names = {
  [HEAD]:'Femoral head cartilage', [SOCKET]:'Acetabular cartilage', [BAND]:'Iliotibial band',
  'Hip bone':'Hip bone', Femur:'Proximal femur', [CAPSULE]:'Hip joint capsule',
};
const capsularLigament = /iliofemoral|pubofemoral|ischiofemoral|zona orbicularis/i;
const jointLigament = /head of femur|acetabular ligament/i;
const colors = {
  bone:'#deceb3', cartilage:'#b4d1cc', labrum:'#ba8f89', capsule:'#bbb293',
  band:'#dbceae', fascia:'#c8bda6', muscle:'#a65348', tendons:'#dec9a5',
  nerves:'#d5ae53', vessels:'#a84846', ligaments:'#d2ba98', bursae:'#879ca9',
};
const specificColors = {
  [MAXIMUS]:'#a34b42', [MEDIUS]:'#bb6556', [MINIMUS]:'#be7162', 'Tensor fasciae latae':'#a05c50',
  'Piriformis muscle':'#c77b65', 'Superior gemellus muscle':'#be675c',
  'Inferior gemellus muscle':'#b56353', 'Obturator internus':'#b27567',
  'Quadratus femoris muscle':'#a65047', [SOCKET]:'#9dbebc', [HEAD]:'#c3d9d4', 'Sacrum':'#c8b8a3',
};
export function describeSurface(extras) {
  const {label, regionGroup} = extras;
  let layer = sourceLayer[extras.layer];
  if (label === BAND) layer = 'band';
  else if (label === LABRUM) layer = 'labrum';
  else if (label === CAPSULE) layer = 'capsule';
  else if (regionGroup === 'vessels' || regionGroup === 'bursae' || regionGroup === 'nerves') layer = regionGroup;
  else if (/burs(a|ae)/i.test(label)) layer = 'bursae';
  const glutealTendon = label === MEDIUS || label === MINIMUS;
  const title = names[label] || label.replace(/ muscle$/, '');
  const color = /vein/i.test(label) ? '#597d8d' : specificColors[label] || colors[layer];
  return {label, regionGroup, layer, title, color:color || '#c8ac94', glutealTendon};
}

export function profileContains(profile, {label, regionGroup, layer}) {
  const bones = ['Hip bone','Femur','Sacrum'];
  if (profile === 'joint') {
    return ['Hip bone','Femur',HEAD,SOCKET,LABRUM,CAPSULE].includes(label) || jointLigament.test(label) || capsularLigament.test(label);
  }
  if (profile === 'overview') {
    return [...bones, HEAD, SOCKET, LABRUM, MEDIUS, MINIMUS, BAND, CAPSULE, 'Tensor fasciae latae'].includes(label)
      || (layer === 'bursae' && /trochanteric/i.test(label));
  }
  if (bones.includes(label) || [HEAD,SOCKET,LABRUM].includes(label)) return true;
  if (profile === 'surface') {
    return ['muscle','fascia','band','tendons','bursae'].includes(layer) && !['Psoas major','Psoas minor'].includes(label);
  }
  return [MINIMUS,MEDIUS].includes(label) || regionGroup === 'rotators'
    || ['nerves','vessels'].includes(layer) || ['Sacrospinous ligament','Sacrotuberal ligament'].includes(label)
    || (layer === 'bursae' && /^Trochanteric bursa/.test(label));
}

export function visibleSurface(profile, info, layers) {
  const {layer, label, regionGroup, glutealTendon} = info;
  if (profile === 'surface' && label === 'Fascia lata') return layers.envelope;
  if (profile === 'deep' && label === MEDIUS && !layers.medius) return false;
  if (profile === 'deep' && regionGroup === 'rotators' && !layers.rotators) return false;
  if (glutealTendon) return layers.muscle || (layers.tendons && profile !== 'surface');
  // Cartilage deep to the muscles follows the bone control outside the joint pane.
  if (['surface','deep'].includes(profile) && ['cartilage','labrum'].includes(layer)) return layers.bone;
  if (profile === 'deep' && layer === 'ligaments') return layers.bone;
  if (profile === 'joint' && label === CAPSULE) return layers.capsule;
  if (profile === 'joint' && label === 'Ligament of head of femur') return layers.ligaments && !layers.separate;
  if (profile === 'joint' && capsularLigament.test(label)) return layers.ligaments && !layers.separate;
  return Boolean(layers[layer]);
}

export function atlasMaterial(info, uniforms, colorsBaked, source) {
  const {label, layer, color, glutealTendon} = info;
  const fascia = layer === 'fascia';
  const cartilage = layer === 'cartilage';
  const bursa = layer === 'bursae';
  const muscle = layer === 'muscle';
  const material = new THREE.MeshPhysicalMaterial({
    color:source.map ? '#ffffff' : color, vertexColors:colorsBaked, metalness:0,
    map:source.map, normalMap:source.normalMap, normalScale:source.normalScale,
    roughnessMap:source.roughnessMap, aoMap:source.aoMap,
    roughness: cartilage ? .30 : bursa ? .37 : muscle ? .54 : .68,
    clearcoat: cartilage ? .29 : bursa ? .22 : muscle ? .09 : .025,
    clearcoatRoughness: cartilage ? .22 : .43,
    sheen: muscle || fascia ? .34 : 0, sheenRoughness:.72, sheenColor:'#bba9a5',
    transparent: fascia || bursa, opacity:label === 'Fascia lata' ? .12 : fascia ? .44 : bursa ? .85 : 1,
    depthWrite:!fascia, side:THREE.DoubleSide,
  });
  material.forceSinglePass = true;
  if (!glutealTendon && !cartilage && layer !== 'band') return material;
  material.customProgramCacheKey = () => 'hip-surface-v3/'+label;
  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying float vTendon; attribute float hipInsertion;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvTendon = hipInsertion;');
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>',
      '#include <common>\nvarying float vTendon; uniform float showBody, showTendon, jointColor, tendonColor, bandColor;');
    shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', '#include <color_fragment>\n'+
      (glutealTendon ? `if ((showBody < .5 && vTendon < .5) || (showTendon < .5 && vTendon >= .5)) discard;
        diffuseColor.rgb = mix(diffuseColor.rgb, mix(vec3(.75,.60,.43),vec3(.91,.38,.15),tendonColor),vTendon*.9);` : '') +
      (cartilage ? 'diffuseColor.rgb = mix(diffuseColor.rgb,vec3(.18,.66,.58),jointColor*.7);' : '') +
      (layer === 'band' ? 'diffuseColor.rgb = mix(diffuseColor.rgb,vec3(.86,.45,.17),bandColor*.75);' : ''));
  };
  return material;
}

// Publisher texture coordinates wrap along the original surfaces. Planar cut
// faces added by extraction have no measured cross-section texture or normals.
export function shadeSection(geometry, sections, pivotY, info, material) {
  if (!sections?.length) return;
  const {position,normal} = geometry.attributes;
  const flags = new Float32Array(position.count);
  for (let i=0; i<flags.length; i++) {
    if (Math.abs(normal.getY(i)) < .99999) continue;
    if (sections.some(s => Math.abs(s.yM - pivotY - position.getY(i)) < .00001)) flags[i] = 1;
  }
  if (!flags.some(Boolean)) return;
  geometry.setAttribute('atlasSection',new THREE.BufferAttribute(flags,1));
  const color = new THREE.Color(info.color).multiplyScalar(.80);
  const before = material.onBeforeCompile.bind(material);
  const key = material.customProgramCacheKey();
  material.customProgramCacheKey = () => key+'/section';
  material.onBeforeCompile = (shader, renderer) => {
    before(shader,renderer);
    shader.uniforms.sectionColor = {value:color};
    shader.vertexShader = shader.vertexShader.replace('#include <common>',
      '#include <common>\nattribute float atlasSection; varying float vSection;')
      .replace('#include <begin_vertex>','#include <begin_vertex>\nvSection = atlasSection;');
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>',
      '#include <common>\nvarying float vSection; uniform vec3 sectionColor;')
      .replace('#include <roughnessmap_fragment>',
        '#include <roughnessmap_fragment>\nif (vSection > .99) { diffuseColor.rgb = sectionColor; roughnessFactor = .85; }')
      .replace('#include <normal_fragment_maps>',
        'vec3 sectionNormal = normal;\n#include <normal_fragment_maps>\nif (vSection > .99) normal = sectionNormal;');
  };
}
