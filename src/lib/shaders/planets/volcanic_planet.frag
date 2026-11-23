#ifdef GL_ES
precision highp float;
#endif

uniform vec2 resolution;
uniform float seed;
uniform float render_background;
// Full quality lights the relief of the crust. Reduced quality lights the smooth globe.
uniform float bump_detail;
uniform float planet_radius;
uniform vec3 light_direction;


uniform float has_rings;
uniform float ring_angle;
uniform float ring_tilt;
uniform vec3 ring_color;

varying vec2 vUvs;

float inverseLerp(float v, float minValue, float maxValue) {
  return (v - minValue) / (maxValue - minValue);
}

float remap(float v, float inMin, float inMax, float outMin, float outMax) {
  float t = inverseLerp(v, inMin, inMax);
  return mix(outMin, outMax, t);
}

float saturate(float x) {
  return clamp(x, 0.0, 1.0);
}

/**
 * OpenSimplex functions by KdotJPG
 * https://github.com/KdotJPG/OpenSimplex2
 */

// Inspired by Stefan Gustavson's noise
vec4 permute(vec4 t) {
    return t * (t * 34.0 + 133.0);
}

// Gradient set is a normalized expanded rhombic dodecahedron
vec3 grad(float hash) {

    // Random vertex of a cube, +/- 1 each
    vec3 cube = mod(floor(hash / vec3(1.0, 2.0, 4.0)), 2.0) * 2.0 - 1.0;

    // Random edge of the three edges connected to that vertex
    // Also a cuboctahedral vertex
    // And corresponds to the face of its dual, the rhombic dodecahedron
    vec3 cuboct = cube;
    int index = int(hash / 16.0);
    if (index == 0) cuboct.x = 0.0;
    else if (index == 1) cuboct.y = 0.0;
    else cuboct.z = 0.0;

    // In a funky way, pick one of the four points on the rhombic face
    float type = mod(floor(hash / 8.0), 2.0);
    vec3 rhomb = (1.0 - type) * cube + type * (cuboct + cross(cube, cuboct));

    // Expand it so that the new edges are the same length
    // as the existing ones
    vec3 grad = cuboct * 1.22474487139 + rhomb;

    // To make all gradients the same length, we only need to shorten the
    // second type of vector. We also put in the whole noise scale constant.
    // The compiler should reduce it into the existing floats. I think.
    grad *= (1.0 - 0.042942436724648037 * type) * 32.80201376986577;

    return grad;
}

// BCC lattice split up into 2 cube lattices
vec4 openSimplex2Base(vec3 X) {

    // First half-lattice, closest edge
    vec3 v1 = floor(X + 0.5);
    vec3 d1 = X - v1;
    vec3 score1 = abs(d1);
    vec3 dir1 = step(max(score1.yzx, score1.zxy), score1);
    vec3 v2 = v1 + dir1 * sign(d1);
    vec3 d2 = X - v2;

    // Second half-lattice, closest edge
    vec3 X2 = X + 144.5;
    vec3 v3 = floor(X2 + 0.5);
    vec3 d3 = X2 - v3;
    vec3 score2 = abs(d3);
    vec3 dir2 = step(max(score2.yzx, score2.zxy), score2);
    vec3 v4 = v3 + dir2 * sign(d3);
    vec3 d4 = X2 - v4;

    // Gradient hashes for the four points, two from each half-lattice
    vec4 hashes = permute(mod(vec4(v1.x, v2.x, v3.x, v4.x), 289.0));
    hashes = permute(mod(hashes + vec4(v1.y, v2.y, v3.y, v4.y), 289.0));
    hashes = mod(permute(mod(hashes + vec4(v1.z, v2.z, v3.z, v4.z), 289.0)), 48.0);

    // Gradient extrapolations & kernel function
    vec4 a = max(0.5 - vec4(dot(d1, d1), dot(d2, d2), dot(d3, d3), dot(d4, d4)), 0.0);
    vec4 aa = a * a; vec4 aaaa = aa * aa;
    vec3 g1 = grad(hashes.x); vec3 g2 = grad(hashes.y);
    vec3 g3 = grad(hashes.z); vec3 g4 = grad(hashes.w);
    vec4 extrapolations = vec4(dot(d1, g1), dot(d2, g2), dot(d3, g3), dot(d4, g4));

    // Derivatives of the noise
    vec4 m = aa * a * extrapolations;
    vec3 derivative = -8.0 * (d1 * m.x + d2 * m.y + d3 * m.z + d4 * m.w)
        + (g1 * aaaa.x + g2 * aaaa.y + g3 * aaaa.z + g4 * aaaa.w);

    // Return it all as a vec4
    return vec4(derivative, dot(aaaa, extrapolations));
}

// Use this if you don't want Z to look different from X and Y
vec4 openSimplex2_Conventional(vec3 X) {

    // Rotate around the main diagonal. Not a skew transform.
    vec4 result = openSimplex2Base(dot(X, vec3(2.0/3.0)) - X);
    return vec4(dot(result.xyz, vec3(2.0/3.0)) - result.xyz, result.w);
}

// Use this if you want to show X and Y in a plane, then use Z for time, vertical, etc.
vec4 openSimplex2_ImproveXY(vec3 X) {

    // Rotate so Z points down the main diagonal. Not a skew transform.
    mat3 orthonormalMap = mat3(
        0.788675134594813, -0.211324865405187, -0.577350269189626,
        -0.211324865405187, 0.788675134594813, -0.577350269189626,
        0.577350269189626, 0.577350269189626, 0.577350269189626);

    vec4 result = openSimplex2Base(orthonormalMap * X);
    return vec4(result.xyz * orthonormalMap, result.w);
}

// End OpenSimplex functions

// The MIT License
// Copyright © 2013 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// https://www.youtube.com/c/InigoQuilez
// https://iquilezles.org/
//
// https://www.shadertoy.com/view/Xsl3Dl
vec3 hash3( vec3 p ) // replace this by something better
{
	p = vec3( dot(p,vec3(127.1,311.7, 74.7)),
            dot(p,vec3(269.5,183.3,246.1)),
            dot(p,vec3(113.5,271.9,124.6)));

	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float fbm(vec3 p, int octaves, float persistence, float lacunarity, float exponentiation) {
  float amplitude = 0.5;
  float frequency = 1.0;
  float total = 0.0;
  float normalization = 0.0;

  for (int i = 0; i < 16; ++i) {
    if (i >= octaves) break;
    float noiseValue = openSimplex2_Conventional(p * frequency).w;
    total += noiseValue * amplitude;
    normalization += amplitude;
    amplitude *= persistence;
    frequency *= lacunarity;
  }

  total /= normalization;
  total = total * 0.5 + 0.5;
  total = pow(max(0.0, total), exponentiation);

  return total;
}

vec3 GenerateGridStars(
    vec2 pixelCoords, float starRadius, float cellWidth,
    float variant, bool twinkle) {

  float seedVariant = clamp((seed + variant) / 100.0, 0.0, 1.0);
  vec2 cellCoords = (fract(pixelCoords / cellWidth) - 0.5 + seedVariant) * cellWidth;
  vec2 cellID = floor(pixelCoords / cellWidth);
  vec3 cellHashValue = hash3(vec3(cellID, 0.0));

  float starBrightness = saturate(cellHashValue.z);
  vec2 starPosition = vec2(seedVariant);
  starPosition += cellHashValue.xy * (cellWidth * 0.5 - starRadius * 4.0 + seedVariant);
  float distToStar = length(cellCoords - starPosition);
  float glow = exp(-2.0 * distToStar / starRadius);

  if (twinkle) {
    float noiseSample = openSimplex2_Conventional(vec3(cellID, seedVariant)).w;
    float twinkleSize = (
        remap(noiseSample, -1.0, 1.0, 1.0, 0.1) * starRadius * 6.0);
    vec2 absDist = abs(cellCoords - starPosition);
    float twinkleValue = smoothstep(starRadius * 0.25, 0.0, absDist.y) *
        smoothstep(twinkleSize, 0.0, absDist.x);
    twinkleValue += smoothstep(starRadius * 0.25, 0.0, absDist.x) *
        smoothstep(twinkleSize, 0.0, absDist.y);
    glow += twinkleValue;
  }

  return vec3(glow * starBrightness);
}

vec3 GenerateStars(vec2 pixelCoords) {
  vec3 stars = vec3(0.0);

  float size = 4.0;
  float cellWidth = 700.0;
  for (int i = 0; i < 3; i++) {
    stars += GenerateGridStars(pixelCoords, size, cellWidth, float(i), true);
    size *= 0.5;
    cellWidth *= 0.35;
  }

  for (int i = 3; i < 5; i++) {
    stars += GenerateGridStars(pixelCoords, size, cellWidth, float(i), false);
    size *= 0.5;
    cellWidth *= 0.35;
  }

  return stars;
}

float sdfCircle(vec2 p, float r) {
  return length(p) - r;
}

// Distance to the nearest volcanic centre and the gaps to the next two centres.
// Sample on the sphere so calderas stay round at the poles and rifts have no map seam.
vec3 volcanicProvince(vec3 direction) {
  float nearest = 4.0;
  float second = 4.0;
  float third = 4.0;
  for (int i = 0; i < 28; ++i) {
    float n = float(i);
    float y = 1.0 - 2.0 * (n + 0.5) / 28.0;
    float longitude = n * 2.399963 + seed;
    vec2 parallel = sqrt(1.0 - y * y) * vec2(cos(longitude), sin(longitude));
    float distanceSquared = 2.0 - 2.0 * dot(direction, vec3(parallel.x, y, parallel.y));
    third = min(third, max(second, distanceSquared));
    second = min(second, max(nearest, distanceSquared));
    nearest = min(nearest, distanceSquared);
  }
  float radius = sqrt(max(0.0, nearest));
  return vec3(radius, sqrt(max(0.0, second)) - radius, sqrt(max(0.0, third)) - radius);
}

vec3 volcanicAtmosphere(float type) {
  if (type < 0.25) return vec3(0.50, 0.21, 0.09);
  if (type < 0.50) return vec3(0.65, 0.44, 0.15);
  if (type < 0.75) return vec3(0.29, 0.18, 0.17);
  return vec3(0.57, 0.20, 0.10);
}

vec3 volcanicSurface(vec3 normal, float type, vec3 atmosphere) {
  vec3 offset = seed * vec3(0.47, 0.83, 0.31);
  vec3 p = normal * 2.1 + offset;
  vec3 warp = vec3(openSimplex2_Conventional(p).w,
                   openSimplex2_Conventional(p + 17.0).w,
                   openSimplex2_Conventional(p - 9.0).w);
  vec3 bent = normal + 0.13 * warp;
  vec3 terrain = bent * 2.2 + offset;
  float highland = fbm(terrain, 4, 0.44, 2.1, 1.0);
  float folds = fbm(terrain * vec3(3.2, 8.0, 2.1), 3, 0.5, 2.1, 1.0);
  float ridge = 1.0 - smoothstep(0.0, 0.19, abs(folds - 0.48));

  // A low-frequency winding channel. Convert the noise value to approximate surface
  // distance using its tangent gradient. Keep the sign for pixel coverage across both banks.
  vec4 course = openSimplex2_Conventional(terrain * 0.84 + 8.0);
  vec3 slope = course.xyz - normal * dot(normal, course.xyz);
  float riverDistance = course.w / max(0.1, 1.85 * length(slope));
  float riverHalfWidth = 0.0024;
  // Fill low, broad depressions. Fine folds roughen their shores without turning
  // a long, narrow valley into a river-width band of lake pixels.
  float basinFloor = 0.72 * openSimplex2_Conventional(terrain - 0.12 * warp).w
                  + 0.28 * openSimplex2_Conventional(terrain * 2.2 + 4.5).w;
  float lakeDepth = (-0.45 - basinFloor) * 0.30;

  vec3 lowRock, highRock, hot, ember;
  float height, deposits;
  if (type < 0.25) {
    // Overlapping basalt shelves. Ash picks out the ledges in successive cooled flows.
    float shelf = smoothstep(0.40, 0.45, highland) + smoothstep(0.52, 0.57, highland);
    height = 0.30 * shelf + 0.28 * highland + 0.12 * ridge;
    deposits = 0.28 * shelf + 0.32 * ridge + 0.12 * folds;
    lowRock = vec3(0.022, 0.018, 0.021);
    highRock = vec3(0.32, 0.28, 0.26);
    hot = vec3(1.0, 0.46, 0.046);
    ember = vec3(0.45, 0.042, 0.005);
  } else if (type < 0.50) {
    // Flat-floored calderas, raised rims and yellow mineral aprons on the outer slopes.
    vec3 province = volcanicProvince(normalize(normal + 0.065 * warp));
    float crater = province.x - (0.135 + 0.075 * warp.z);
    float rim = 1.0 - smoothstep(0.0, 0.085, abs(crater));
    float apron = 1.0 - smoothstep(0.03, 0.22, crater);
    height = 0.39 * rim + 0.24 * apron + 0.27 * highland + 0.10 * ridge;
    deposits = 0.56 * apron + 0.25 * ridge + 0.35 * highland;
    lakeDepth = max(lakeDepth * 0.65, -crater - 0.035 - 0.035 * warp.x);
    riverHalfWidth = 0.0018;
    lowRock = vec3(0.14, 0.074, 0.037);
    highRock = vec3(0.82, 0.64, 0.25);
    hot = vec3(1.0, 0.27, 0.022);
    ember = vec3(0.48, 0.034, 0.005);
  } else if (type < 0.75) {
    // Isolated remnant massifs over a dark, solidified plain. Exposed crests catch pale ash.
    float massif = smoothstep(0.43, 0.66, highland);
    height = 0.20 * highland + massif * (0.36 + 0.44 * ridge);
    deposits = 0.12 * folds + massif * (0.25 + 0.60 * ridge);
    lakeDepth = (-0.54 - basinFloor) * 0.30;
    riverHalfWidth = 0.0016;
    lowRock = vec3(0.019, 0.021, 0.025);
    highRock = vec3(0.39, 0.40, 0.42);
    hot = vec3(0.72, 0.11, 0.014);
    ember = vec3(0.24, 0.014, 0.004);
  } else {
    // Angular rafts of uplifted crust. Lava follows the faults and floods their low junctions.
    vec3 province = volcanicProvince(normalize(normal + 0.066 * warp));
    riverDistance = 0.5 * province.y;
    riverHalfWidth = 0.0026;
    // A three-way junction forms a pocket, while the fault between two plates stays thin.
    lakeDepth = max((-0.53 - basinFloor) * 0.28, (0.055 + 0.04 * warp.x - province.z) * 0.8);
    float scarp = smoothstep(0.025, 0.075, riverDistance);
    height = 0.38 * scarp + 0.37 * highland + 0.22 * ridge;
    deposits = 0.37 * scarp + 0.37 * ridge + 0.30 * highland;
    lowRock = vec3(0.052, 0.025, 0.024);
    highRock = vec3(0.33, 0.26, 0.23);
    hot = vec3(1.0, 0.32, 0.026);
    ember = vec3(0.55, 0.042, 0.006);
  }

  // Differential erosion along the shore: lava reaches inland through softer seams,
  // while resistant ribs project into the basin. Work at the basin margin so
  // caldera floors stay filled and the fine river mask keeps its orbital scale.
  vec3 strata = bent * vec3(5.0, 10.0, 4.0) + offset + 0.35 * warp;
  float bedding = openSimplex2_Conventional(strata).w;
  float joint = openSimplex2_Conventional(strata * 1.9 + 12.0).w;
  float exposedBed = bedding + 0.32 * joint;
  float inlet = 1.0 - smoothstep(-0.55, 0.10, exposedBed);
  float headland = smoothstep(0.02, 0.62, exposedBed) * (0.65 + 0.35 * ridge);
  float shore = 1.0 - smoothstep(0.018, 0.078, abs(lakeDepth));
  float reach = 0.70 + 0.30 * warp.z;
  float wear = shore * reach * (0.043 * inlet - 0.032 * headland);
  lakeDepth += wear;
  height -= 1.6 * wear;

  // Channel widths are fractions of the planet radius. Even the widest is less than
  // one pixel across on a 320 px disc. Only active stretches expose enough lava to see.
  riverHalfWidth *= smoothstep(-0.06, 0.66, warp.y - 0.35 * (highland - 0.45));
  float footprint = max(fwidth(riverDistance), 0.65 * length(fwidth(normal)));
  footprint = max(footprint, 0.0001);
  // Integrate the strip over a pixel. An unresolved channel contributes in proportion
  // to its width, including when reduced quality renders fewer fragments.
  float nearBank = clamp(0.5 + (abs(riverDistance) + riverHalfWidth) / footprint, 0.0, 1.0);
  float farBank = clamp(0.5 + (abs(riverDistance) - riverHalfWidth) / footprint, 0.0, 1.0);
  float riverCoverage = nearBank - farBank;
  float lakePixel = max(fwidth(lakeDepth), 0.001);
  float lakeCoverage = smoothstep(-0.5 * lakePixel, 0.5 * lakePixel, lakeDepth);

  vec3 light = normalize(light_direction);
  vec3 surfaceNormal = normal;
  if (bump_detail > 0.5) {
    // Derive relief from the actual shelves, rims and scarps used to colour this seed.
    vec3 rockSurface = normal * (1.0 + 0.016 * height);
    vec3 relief = cross(dFdx(rockSurface), dFdy(rockSurface));
    surfaceNormal = normalize(normal + relief / max(length(relief), 0.000001));
  }
  float daylight = smoothstep(-0.12, 0.24, dot(light, normal));
  float sun = daylight * max(0.0, dot(light, surfaceNormal));
  float grain = fbm(terrain * 21.0, 2, 0.5, 2.0, 1.0);
  vec3 rock = mix(lowRock, highRock, clamp(deposits * deposits, 0.0, 1.0));
  // Pale striations collect on exposed high ground, leaving expanses of bare dark rock.
  float ash = smoothstep(0.34, 0.67, highland + 0.15 * warp.y);
  rock *= mix(0.58, 1.25, ash);
  rock += highRock * ash * smoothstep(0.46, 0.72, folds) * 0.19;
  rock *= (0.70 + 0.65 * grain) * (0.025 + 0.98 * sun);
  float bankGlow = exp(-max(0.0, -lakeDepth) * 220.0);
  rock += ember * 0.13 * bankGlow;

  // Cooler banks outline a broad incandescent interior. Streaks and cooling rafts
  // are confined to the liquid so they cannot break up the shoreline.
  float interior = smoothstep(0.0, 0.046, lakeDepth);
  float streak = smoothstep(0.33, 0.68, folds);
  vec3 lava = mix(ember, hot, (0.08 + 0.92 * interior) * (0.25 + 0.75 * streak));
  lava *= (0.75 + 0.25 * daylight) * (1.0 - 0.33 * interior * smoothstep(0.55, 0.69, grain));
  vec3 color = mix(rock, mix(ember, hot, 0.25), riverCoverage);
  color = mix(color, lava, lakeCoverage);
  float rimLight = pow(1.0 - max(0.0, normal.z), 4.0) * max(0.0, dot(light, normal));
  return mix(color, atmosphere, 0.75 * rimLight);
}

vec3 DrawPlanet(vec2 pixelCoords, vec3 color, float planetRadius) {
  float d = sdfCircle(pixelCoords, planetRadius);
  // Surface derivatives need the adjacent fragments even when they land just beyond
  // the limb. Shading this margin makes tiny planets reproducible in system previews.
  float derivativeMargin = length(fwidth(pixelCoords));
  if (d >= planetRadius * 0.05 + 24.0) return color;

  float type = fract(seed * 0.77341);
  vec3 atmosphere = volcanicAtmosphere(type);
  if (d <= derivativeMargin) {
    vec2 xy = pixelCoords / planetRadius;
    vec3 normal = vec3(xy, sqrt(max(0.0, 1.0 - dot(xy, xy))));
    vec3 planetColor = volcanicSurface(normalize(normal), type, atmosphere);
    color = mix(color, planetColor, 1.0 - smoothstep(-1.0, 0.0, d));
  }

  // Atmosphere glow (halo outside the planet)
  float atmosphereAmount = planetRadius * 0.05;
  float atmosphereRadius = planetRadius + atmosphereAmount;

  if (d >= -1.0) {
    float x = pixelCoords.x / atmosphereRadius;
    float y = pixelCoords.y / atmosphereRadius;
    float z = sqrt(max(0.0, 1.0 - x * x - y * y));
    vec3 normal = vec3(x, y, z);

    float lighting = dot(normal, normalize(light_direction));
    lighting = smoothstep(-0.15, 1.0, lighting);

    float glowDegree = -0.1;
    glowDegree *= 512.0 / resolution.x;

    vec3 glowColor = atmosphere * exp(glowDegree * d * d) * lighting * 0.65;
    color += glowColor;
  }

  return color;
}


vec3 DrawRings(vec2 pixelCoords, vec3 color, float planetRadius, bool front) {
  if (has_rings < 0.5) return color;

  float sizeSeed = fract(seed * 0.123);
  float innerR = planetRadius * (1.1 + sizeSeed * 0.8); 
  float outerR = innerR + planetRadius * (0.2 + fract(seed * 0.765) * 1.5);

  float s = sin(ring_angle);
  float c = cos(ring_angle);
  mat2 rot = mat2(c, -s, s, c);

  vec2 rP = rot * pixelCoords;
  if (front && rP.y < 0.0) return color;
  if (!front && rP.y >= 0.0) return color;

  vec2 dp = vec2(1.5 / resolution.x, 1.5 / resolution.y);
  
  float bandTotal = 0.0;
  float alphaTotal = 0.0;
  float validSamples = 0.0;
  vec3 colorTotal = vec3(0.0);

  for (float dx = -0.5; dx <= 0.5; dx += 1.0) {
    for (float dy = -0.5; dy <= 0.5; dy += 1.0) {
       vec2 jp = rot * (pixelCoords + vec2(dx, dy) * dp);
       jp.y /= max(0.05, ring_tilt);
       float jd = length(jp);
       if (jd > innerR && jd < outerR) {
           float jt = (jd - innerR) / (outerR - innerR);
           
           // Lower frequencies significantly to create massive, continuous bands
           float f_big = 4.0 + sizeSeed * 6.0;
           float noiseBig = sin(jt * f_big + seed * 10.0);
           
           float f_med = 15.0 + fract(sizeSeed * 2.0) * 10.0;
           float noiseMed = sin(jt * f_med - seed * 5.0);
           
           float f_small = 40.0;
           float noiseSmall = sin(jt * f_small + seed);
           
           // Extremely wide coverage. By using -0.8 to -0.2, the ring is solid 1.0 
           // for over 80% of the sine wave, only dipping in the deep valleys.
           float bigRing = smoothstep(-0.8, -0.2, noiseBig);
           
           // Medium bands don't go exactly to 0, leaving a translucent floor 
           // instead of empty space.
           float medRing = mix(0.4, 1.0, smoothstep(-0.2, 0.6, noiseMed));
           
           float jband = bigRing * medRing;
           // Add small detail bands
           jband += smoothstep(0.6, 1.0, abs(noiseSmall)) * 0.15 * bigRing;
           
           bandTotal += jband;
           alphaTotal += clamp(jband, 0.0, 1.0) * 0.9;
           validSamples += 1.0;
           
           vec3 baseCol = ring_color;
           // Darken using the medium bands to create structural depth
           baseCol *= mix(0.6, 1.0, smoothstep(0.0, 1.0, noiseMed));
           
           // Subtle hue shifts using the noise channels
           vec3 hueShift = vec3(noiseBig * 0.05, noiseMed * 0.05, (noiseBig * noiseMed) * 0.08);
           vec3 sampleColor = clamp(baseCol + hueShift, 0.0, 1.0);
           colorTotal += sampleColor;
       }
    }
  }

  if (validSamples > 0.0) {
    float band = bandTotal / validSamples;
    float alpha = alphaTotal / validSamples;
    vec3 finalRingColor = colorTotal / validSamples;
    
    vec2 p = rot * pixelCoords;
    p.y /= max(0.05, ring_tilt);
    float d = length(p);
    
    float edge = smoothstep(innerR, innerR + planetRadius * 0.01, d) * 
                 (1.0 - smoothstep(outerR - planetRadius * 0.01, outerR, d));
    
    float shadow = 1.0;
    vec2 ldXY = normalize(light_direction.xy);
    float dotLight = dot(normalize(pixelCoords), ldXY);
    if (dotLight < 0.0) {
      float lineDist = abs(pixelCoords.x * ldXY.y - pixelCoords.y * ldXY.x);
      if (lineDist < planetRadius) {
         shadow = mix(1.0, 0.05, smoothstep(planetRadius, planetRadius * 0.8, lineDist) * smoothstep(0.0, -0.4, dotLight));
      }
    }
    
    float highlight = 1.0 + smoothstep(0.4, 1.0, band) * 0.25 * max(0.0, dotLight);
    return mix(color, finalRingColor * shadow * highlight, alpha * edge);
  }
  return color;
}

void main() {
  vec2 pixelCoords = (vUvs - 0.5) * resolution;

  // render_background is honoured rather than ignored: the scene draws the background once, in its
  // own pass, from the star positions both backends share. A body plane that generated its own
  // starfield would paint a different sky over that one — and, in a star system, over its
  // neighbours as well.
  vec3 color = render_background > 0.5 ? GenerateStars(pixelCoords) : vec3(0.0);
  color = DrawRings(pixelCoords, color, planet_radius, false);
  color = DrawPlanet(pixelCoords, color, planet_radius);
  color = DrawRings(pixelCoords, color, planet_radius, true);

  vec3 finalColor = pow(max(vec3(0.0), color), vec3(1.0 / 2.2));

  // Nothing was drawn at this pixel, so let the background through instead of covering it with an
  // opaque black square. The threshold is below one 8-bit step; the planet's night side sits an
  // order of magnitude above it and still occludes what is behind it.
  if (render_background < 0.5 && max(finalColor.r, max(finalColor.g, finalColor.b)) < 0.004) {
    discard;
  }

  gl_FragColor = vec4(finalColor, 1.0);
}
