// STL
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// extern
#include <omp.h>

//////////////////////////////////////////

// base computation type
using Real = float;

//////////////////////////////////////////

// global constants
constexpr Real PI = 3.14159265358979323846;
constexpr Real PI2 = 2.0 * PI;
constexpr Real INV_PI = 1.0 / PI;
constexpr Real INV_PI2 = 1.0 / PI2;
constexpr Real ONE_MINUS_EPS = 1.0 - std::numeric_limits<Real>::epsilon();

//////////////////////////////////////////

// 3-dimensional vector
class Vec3 {
 public:
  Real x;
  Real y;
  Real z;

  Vec3() { x = y = z = 0; }
  Vec3(Real _x) { x = y = z = _x; }
  Vec3(Real _x, Real _y, Real _z) : x(_x), y(_y), z(_z) {}

  Vec3& operator+=(const Vec3& v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  Vec3& operator+=(Real k) {
    x += k;
    y += k;
    z += k;
    return *this;
  }
  Vec3& operator-=(const Vec3& v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  Vec3& operator-=(Real k) {
    x -= k;
    y -= k;
    z -= k;
    return *this;
  }
  Vec3& operator*=(const Vec3& v) {
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
  }
  Vec3& operator*=(Real k) {
    x *= k;
    y *= k;
    z *= k;
    return *this;
  }
  Vec3& operator/=(const Vec3& v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
  }
  Vec3& operator/=(Real k) {
    x /= k;
    y /= k;
    z /= k;
    return *this;
  }
};

// Vec3 operators
Vec3 operator+(const Vec3& v1, const Vec3& v2) {
  return Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
Vec3 operator+(const Vec3& v, Real k) {
  return Vec3(v.x + k, v.y + k, v.z + k);
}
Vec3 operator+(Real k, const Vec3& v) {
  return Vec3(k + v.x, k + v.y, k + v.z);
}

Vec3 operator-(const Vec3& v1, const Vec3& v2) {
  return Vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
Vec3 operator-(const Vec3& v, Real k) {
  return Vec3(v.x - k, v.y - k, v.z - k);
}
Vec3 operator-(Real k, const Vec3& v) {
  return Vec3(k - v.x, k - v.y, k - v.z);
}

Vec3 operator*(const Vec3& v1, const Vec3& v2) {
  return Vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}
Vec3 operator*(const Vec3& v, Real k) {
  return Vec3(v.x * k, v.y * k, v.z * k);
}
Vec3 operator*(Real k, const Vec3& v) {
  return Vec3(k * v.x, k * v.y, k * v.z);
}

Vec3 operator/(const Vec3& v1, const Vec3& v2) {
  return Vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}
Vec3 operator/(const Vec3& v, Real k) {
  return Vec3(v.x / k, v.y / k, v.z / k);
}
Vec3 operator/(Real k, const Vec3& v) {
  return Vec3(k / v.x, k / v.y, k / v.z);
}

Real dot(const Vec3& v1, const Vec3& v2) {
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
Vec3 cross(const Vec3& v1, const Vec3& v2) {
  return Vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z,
              v1.x * v2.y - v1.y * v2.x);
}

Real length(const Vec3& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
Real length2(const Vec3& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

Vec3 normalize(const Vec3& v) { return v / length(v); }

Vec3 worldToLocal(const Vec3& v, const Vec3& lx, const Vec3& ly,
                  const Vec3& lz) {
  return Vec3(dot(v, lx), dot(v, ly), dot(v, lz));
}
Vec3 localToWorld(const Vec3& v, const Vec3& lx, const Vec3& ly,
                  const Vec3& lz) {
  return Vec3(dot(v, Vec3(lx.x, ly.x, lz.x)), dot(v, Vec3(lx.y, ly.y, lz.y)),
              dot(v, Vec3(lx.z, ly.z, lz.z)));
}

std::ostream& operator<<(std::ostream& stream, const Vec3& v) {
  stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return stream;
}

//////////////////////////////////////////

// Ray
class Ray {
 public:
  Vec3 origin;     // origin of ray
  Vec3 direction;  // direction of ray

  static constexpr Real tmin = std::numeric_limits<Real>::epsilon();
  static constexpr Real tmax = std::numeric_limits<Real>::max();

  Ray() {}
  Ray(const Vec3& _origin, const Vec3& _direction)
      : origin(_origin), direction(_direction) {}

  Vec3 operator()(Real t) const { return origin + t * direction; }
};

//////////////////////////////////////////

// PCG32 random number generator

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

typedef struct {
  uint64_t state;
  uint64_t inc;
} pcg32_random_t;

uint32_t pcg32_random_r(pcg32_random_t* rng) {
  uint64_t oldstate = rng->state;
  // Advance internal state
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint32_t rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// random number generator
class Sampler {
 public:
  pcg32_random_t state;

  static constexpr uint64_t PCG32_DEFAULT_STATE = 0x853c49e6748fea9bULL;
  static constexpr uint64_t PCG32_DEFAULT_INC = 0xda3e39cb94b95bdbULL;

  Sampler() {
    state.state = PCG32_DEFAULT_STATE;
    state.inc = PCG32_DEFAULT_INC;
  }
  Sampler(uint64_t seed) { setSeed(seed); }

  // set seed of random number generator
  void setSeed(uint64_t seed) {
    state.state = seed;
    uniformReal();
    uniformReal();
  }

  // return random value in [0, 1]
  Real uniformReal() {
    return std::min(static_cast<Real>(pcg32_random_r(&state) * 0x1p-32),
                    ONE_MINUS_EPS);
  }
};

// sampling utils
Vec3 sampleCosineHemisphere(Real u, Real v, Real& pdf) {
  const Real theta = 0.5 * std::acos(1.0 - 2.0 * u);
  const Real phi = 2 * PI * v;
  const Real y = std::cos(theta);
  pdf = y / PI;
  return Vec3(std::cos(phi) * std::sin(theta), y,
              std::sin(phi) * std::sin(theta));
}

//////////////////////////////////////////

// Film
class Film {
 public:
  uint32_t width;            // width of image in [px]
  uint32_t height;           // height of image in [px]
  const Real width_length;   // physical length in x-direction in [m]
  const Real height_length;  // physical length in y-direction in [m]

  // pixels are row-major array.
  Vec3* pixels;       // an array contains RGB at each pixel
  uint64_t* samples;  // number of samples at each pixel

  Film(uint32_t _width, uint32_t _height, Real _width_length = 0.036,
       Real _height_length = 0.024)
      : width(_width),
        height(_height),
        width_length(_width_length),
        height_length(_height_length) {
    pixels = new Vec3[width * height];
    samples = new uint64_t[width * height];

    // initialize samples
    for (int i = 0; i < width * height; ++i) {
      samples[i] = 0;
    }
  }
  ~Film() {
    delete[] pixels;
    delete[] samples;
  }

  // getter and setter
  Vec3 getPixel(uint32_t i, uint32_t j) const { return pixels[j + width * i]; }
  void setPixel(uint32_t i, uint32_t j, const Vec3& rgb) {
    pixels[j + width * i] = rgb;
  }

  // add RGB to pixel
  void addPixel(uint32_t i, uint32_t j, const Vec3& rgb) {
    pixels[j + width * i] += rgb;
    samples[j + width * i] += 1;
  }

  // output ppm image
  void writePPM(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
      printf("failed to open %s", filename.c_str());
      return;
    }

    // ppm header
    file << "P3" << std::endl;
    file << width << " " << height << std::endl;
    file << "255" << std::endl;

    for (int i = 0; i < width; ++i) {
      for (int j = 0; j < height; ++j) {
        const Vec3& rgb = pixels[j + width * i] / samples[j + width * i];
        const uint32_t R =
            std::clamp(static_cast<uint32_t>(255 * rgb.x), 0u, 255u);
        const uint32_t G =
            std::clamp(static_cast<uint32_t>(255 * rgb.y), 0u, 255u);
        const uint32_t B =
            std::clamp(static_cast<uint32_t>(255 * rgb.z), 0u, 255u);

        file << R << " " << G << " " << B << std::endl;
      }
    }

    file.close();
  }
};

//////////////////////////////////////////

// Camera
// pinhole camera model
class Camera {
 public:
  Vec3 origin;   // origin of camera(position of film center in 3-dimensional
                 // space)
  Vec3 forward;  // forward direction of camera
  Vec3 right;    // right direction of camera
  Vec3 up;       // up direction of camera

  std::shared_ptr<Film> film;  // image sensor of camera

  Real focal_length;  // focal length of pinhole camera

  Camera(const Vec3& _origin, const Vec3& _forward,
         const std::shared_ptr<Film>& _film, Real fov)
      : origin(_origin), forward(normalize(_forward)), film(_film) {
    // create orthonormal basis from forward direction vector
    right = normalize(cross(forward, Vec3(0, 1, 0)));
    up = normalize(cross(right, forward));

    std::cout << "[Camera] origin: " << origin << std::endl;
    std::cout << "[Camera] forward: " << forward << std::endl;
    std::cout << "[Camera] right: " << right << std::endl;
    std::cout << "[Camera] up: " << up << std::endl;

    const Real diagonal_length =
        std::sqrt(film->width_length * film->width_length +
                  film->height_length * film->height_length);
    focal_length = diagonal_length / (2 * std::tan(fov));
  }

  // sample ray from camera
  Ray sampleRay(uint32_t i, uint32_t j, Sampler& sampler) {
    // sample position in pixel with super sampling
    const Real u = film->width_length *
                   (2 * (j + sampler.uniformReal()) - film->width) /
                   film->height;
    const Real v = film->height_length *
                   (2 * (i + sampler.uniformReal()) - film->height) /
                   film->height;
    const Vec3 p_film = origin + u * right + v * up;

    // sample ray
    const Vec3 p_pinhole = origin + focal_length * forward;
    return Ray(origin, normalize(p_pinhole - p_film));
  }
};

//////////////////////////////////////////

// Material
// computation on local coordinate(surface normal is y-axis)
class Material {
 public:
  const Vec3 kd;

  Material(const Vec3& _kd) : kd(_kd) {}

  // BRDF sampling
  Vec3 sampleBRDF(Sampler& sampler, Vec3& direction, Real& pdf_solid) const {
    // sample direction
    direction = sampleCosineHemisphere(sampler.uniformReal(),
                                       sampler.uniformReal(), pdf_solid);

    // compute BRDF
    return INV_PI * kd;
  }
};

//////////////////////////////////////////

// Light
class Light {
 public:
  const Vec3 le;

  Light(const Vec3& _le) : le(_le) {}

  Vec3 Le() const { return le; }
};

//////////////////////////////////////////

// IntersectInfo

// prototype declaration of Primitive
class Primitive;

struct IntersectInfo {
  Real t;          // hit distance
  Vec3 hitPos;     // hit potision
  Vec3 hitNormal;  // surface normal at hit position
  Vec3 dpdu;       // derivative of hit position with u(tangent vector)
  Vec3 dpdv;       // derivative of hit position with v(cotangent vector

  std::shared_ptr<Primitive> hitPrimitive;  // hit primitive

  IntersectInfo() : t(std::numeric_limits<Real>::max()) {}
};

//////////////////////////////////////////

// Sphere
class Sphere {
 public:
  const Vec3 center;  // center of sphere
  const Real radius;  // radius of sphere

  Sphere(const Vec3& _center, Real _radius)
      : center(_center), radius(_radius) {}

  // intersect ray with sphere
  bool intersect(const Ray& ray, IntersectInfo& info) const {
    // solve quadratic equation
    const Real b = dot(ray.direction, ray.origin - center);
    const Real c = length2(ray.origin - center);
    const Real D = b * b - c;
    if (D < 0) return false;

    // choose closer hit distance
    const Real t0 = -b - std::sqrt(D);
    const Real t1 = -b + std::sqrt(D);
    Real t = t0;
    if (t < ray.tmin || t > ray.tmax) {
      t = t1;
      if (t < ray.tmin || t > ray.tmax) {
        return false;
      }
    }

    info.t = t;
    info.hitPos = ray(t);
    info.hitNormal = normalize(info.hitPos - center);
    info.dpdu = normalize(Vec3(-info.hitPos.z, 0, info.hitPos.x));

    // compute local coordinate(phi, theta) of hit position
    Real phi = std::atan2(info.hitPos.z, info.hitPos.x);
    if (phi < 0) phi += PI2;
    const Real theta =
        std::acos(std::clamp(info.hitPos.y, Real(-1.0), Real(1.0)));

    info.dpdv = normalize(Vec3(std::cos(phi) * info.hitPos.y,
                               -radius * std ::sin(theta),
                               std::sin(phi) * info.hitPos.y));

    return true;
  }
};

//////////////////////////////////////////

// Primitive
class Primitive {
 public:
  std::shared_ptr<Sphere> sphere;
  std::shared_ptr<Material> material;
  std::shared_ptr<Light> light;

  Primitive(const std::shared_ptr<Sphere>& _sphere,
            const std::shared_ptr<Material>& _material,
            const std::shared_ptr<Light>& _light)
      : sphere(_sphere), material(_material), light(_light) {}

  bool intersect(const Ray& ray, IntersectInfo& info) const {
    return sphere->intersect(ray, info);
  }

  Vec3 sampleBRDF(const IntersectInfo& info, Sampler& sampler, Vec3& direction,
                  Real& pdf_solid) const {
    Vec3 direction_local;
    const Vec3 BRDF = material->sampleBRDF(sampler, direction_local, pdf_solid);

    // convert direction vector from local to world
    direction =
        localToWorld(direction_local, info.dpdu, info.hitNormal, info.dpdv);

    return BRDF;
  }
};

//////////////////////////////////////////

// Intersector
class Intersector {
 public:
  std::vector<std::shared_ptr<Primitive>> prims;  // primitives

  Intersector() {}
  Intersector(const std::vector<std::shared_ptr<Primitive>>& _prims) {}

  // find closest intersection linearly
  bool intersect(const Ray& ray, IntersectInfo& info) const {
    bool hit = false;
    Real t = ray.tmax;
    for (const auto& prim : prims) {
      IntersectInfo temp_info;
      if (prim->intersect(ray, temp_info)) {
        if (temp_info.t < t) {
          hit = true;

          // set intersect info
          info = temp_info;
          info.hitPrimitive = prim;

          // update minimal hit distance
          t = temp_info.t;
        }
      }
    }
    return hit;
  }
};

//////////////////////////////////////////

// Sky
class Sky {
 public:
  const Vec3 le;

  Sky(const Vec3& _le) : le(_le) {}

  Vec3 Le(const Ray& ray) const { return le; }
};

//////////////////////////////////////////

// Scene
class Scene {
 public:
  const std::shared_ptr<Camera> camera;
  const std::vector<std::shared_ptr<Primitive>> prims;
  const std::shared_ptr<Sky> sky;

  Intersector intersector;

  Scene(const std::shared_ptr<Camera>& _camera,
        const std::vector<std ::shared_ptr<Primitive>>& _prims,
        const std ::shared_ptr<Sky>& _sky)
      : camera(_camera), prims(_prims), sky(_sky) {
    // setup intersector
    intersector = Intersector(prims);
  }

  bool intersect(const Ray& ray, IntersectInfo& info) const {
    return intersector.intersect(ray, info);
  }
};

//////////////////////////////////////////

// Integrator
class Integrator {
 public:
  const uint64_t maxDepth = 100;
  const Real russian_roulette_prob = 0.99;

  Integrator() {}

  // compute given ray's radiance
  Vec3 radiance(const Ray& ray_in, const Scene& scene, Sampler& sampler) const {
    Ray ray = ray_in;
    Vec3 radiance;
    Vec3 throughput(1);

    for (uint64_t depth = 0; depth < maxDepth; depth++) {
      // russian roulette
      if (sampler.uniformReal() >= russian_roulette_prob) return radiance;
      throughput /= russian_roulette_prob;

      // compute intersection with scene
      IntersectInfo info;
      if (scene.intersect(ray, info)) {
        const auto prim = info.hitPrimitive;

        // Le
        radiance += throughput * prim->light->Le();

        // BRDF sampling
        Vec3 next_direction;
        Real pdf_solid;
        const Vec3 BRDF =
            prim->sampleBRDF(info, sampler, next_direction, pdf_solid);
        std::cout << BRDF << std::endl;

        // update throughput
        throughput *= BRDF / pdf_solid;

        // update ray
        ray.direction = next_direction;

      } else {
        radiance += throughput * scene.sky->Le(ray);
      }
    }

    return radiance;
  }
};

//////////////////////////////////////////

class Renderer {
 public:
  const Scene scene;
  const Integrator integrator;
  Sampler sampler;

  Renderer(const Scene& _scene, const Integrator& _integrator,
           const Sampler& _sampler)
      : scene(_scene), integrator(_integrator), sampler(_sampler) {}

  void render(uint64_t n_samples) {
    // #pragma omp parallel for schedule(dynamic, 1)
    for (uint32_t i = 0; i < scene.camera->film->height; ++i) {
      for (uint32_t j = 0; j < scene.camera->film->width; ++j) {
        for (uint64_t k = 0; k < n_samples; ++k) {
          // sample ray
          Ray ray = scene.camera->sampleRay(i, j, sampler);

          // compute radiance
          const Vec3 radiance = integrator.radiance(ray, scene, sampler);

          // add radiance on pixel
          scene.camera->film->addPixel(i, j, radiance);
        }
      }
    }

    // write ppm
    scene.camera->film->writePPM("output.ppm");
  }
};

//////////////////////////////////////////

int main() {
  // parameters
  const uint32_t width = 512;
  const uint32_t height = 512;
  const uint64_t samples = 1;

  // setup image
  const auto film = std::make_shared<Film>(width, height);

  // setup camera
  const auto camera =
      std::make_shared<Camera>(Vec3(0, 1, 1), Vec3(0, 0, -1), film, PI / 2.0);

  // setup primitives
  std::vector<std::shared_ptr<Primitive>> prims;
  prims.push_back(std::make_shared<Primitive>(
      std::make_shared<Sphere>(Vec3(0, -10000, 0), 10000),
      std::make_shared<Material>(Vec3(0.8)), std::make_shared<Light>(Vec3(0))));
  prims.push_back(std::make_shared<Primitive>(
      std::make_shared<Sphere>(Vec3(0, 1, 0), 1),
      std::make_shared<Material>(Vec3(0.2, 0.2, 0.8)),
      std::make_shared<Light>(Vec3(0))));

  // setup sky
  const auto sky = std::make_shared<Sky>(Vec3(0.8));

  // setup scene
  Scene scene(camera, prims, sky);

  // setup sampler
  Sampler sampler;

  // setup integrator
  Integrator integrator;

  // setup renderer
  Renderer renderer(scene, integrator, sampler);

  // render
  renderer.render(samples);

  return 0;
}