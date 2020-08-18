// STL
#include <algorithm>
#include <chrono>
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

class Vec3 {
  // 3-dimensional vector
 public:
  Real x;
  Real y;
  Real z;

  Vec3() { x = y = z = 0; }
  Vec3(Real _x) { x = y = z = _x; }
  Vec3(Real _x, Real _y, Real _z) : x(_x), y(_y), z(_z) {}

  Vec3 operator-() const { return Vec3(-x, -y, -z); }

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

// dot product
Real dot(const Vec3& v1, const Vec3& v2) {
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
// cross product
Vec3 cross(const Vec3& v1, const Vec3& v2) {
  return Vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z,
              v1.x * v2.y - v1.y * v2.x);
}

// length of vector
Real length(const Vec3& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
// squared length of vector
Real length2(const Vec3& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

// normalize vector
Vec3 normalize(const Vec3& v) { return v / length(v); }

// convert world vector to local vector
// lx: x-axis basis vector of local coordinate
// ly: y-axis basis vector of local coordinate
// lz: z-axis basis vector of local coordinate
Vec3 worldToLocal(const Vec3& v, const Vec3& lx, const Vec3& ly,
                  const Vec3& lz) {
  return Vec3(dot(v, lx), dot(v, ly), dot(v, lz));
}

// convert local vector to world vector
// lx: x-axis basis vector of local coordinate
// ly: y-axis basis vector of local coordinate
// lz: z-axis basis vector of local coordinate
Vec3 localToWorld(const Vec3& v, const Vec3& lx, const Vec3& ly,
                  const Vec3& lz) {
  return Vec3(dot(v, Vec3(lx.x, ly.x, lz.x)), dot(v, Vec3(lx.y, ly.y, lz.y)),
              dot(v, Vec3(lx.z, ly.z, lz.z)));
}

// print vector
std::ostream& operator<<(std::ostream& stream, const Vec3& v) {
  stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return stream;
}

//////////////////////////////////////////

class Ray {
  // it represents ray
 public:
  Vec3 origin;     // origin of ray
  Vec3 direction;  // direction of ray

  static constexpr Real tmin = 1e-3;
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

//////////////////////////////////////////

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

// sample direction by cosine weighted hemisphere sampling
Vec3 sampleCosineHemisphere(Real u, Real v, Real& pdf) {
  const Real theta = 0.5 * std::acos(1.0 - 2.0 * u);
  const Real phi = PI2 * v;
  const Real y = std::cos(theta);
  pdf = y * INV_PI;
  return Vec3(std::cos(phi) * std::sin(theta), y,
              std::sin(phi) * std::sin(theta));
}

//////////////////////////////////////////

class Film {
  // it represents image sensor of camera
  // store RGB values on each pixel
 public:
  const uint32_t width;      // width of image in [px]
  const uint32_t height;     // height of image in [px]
  const Real width_length;   // physical length in x-direction in [m]
  const Real height_length;  // physical length in y-direction in [m]

  // pixels are row-major array.
  Vec3* pixels;  // an array contains RGB at each pixel

  Film(uint32_t _width, uint32_t _height, Real _width_length = 0.025,
       Real _height_length = 0.025)
      : width(_width),
        height(_height),
        width_length(_width_length),
        height_length(_height_length) {
    pixels = new Vec3[width * height];
  }
  ~Film() { delete[] pixels; }

  // getter and setter
  Vec3 getPixel(uint32_t i, uint32_t j) const { return pixels[j + width * i]; }
  void setPixel(uint32_t i, uint32_t j, const Vec3& rgb) {
    pixels[j + width * i] = rgb;
  }

  // add RGB to pixel
  void addPixel(uint32_t i, uint32_t j, const Vec3& rgb) {
    pixels[j + width * i] += rgb;
  }

  // divide all pixels by given number of samples
  void divide(uint64_t k) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        pixels[j + width * i] /= k;
      }
    }
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

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        Vec3 rgb = getPixel(i, j);

        // gamma correction
        rgb.x = std::pow(rgb.x, 1 / 2.2);
        rgb.y = std::pow(rgb.y, 1 / 2.2);
        rgb.z = std::pow(rgb.z, 1 / 2.2);

        // convert real to uint
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

class Camera {
  // pinhole camera model
 public:
  Vec3 origin;   // origin of camera(position of film center in
                 // 3-dimensional space)
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

    // compute focal length from fov
    const Real diagonal_length =
        std::sqrt(film->width_length * film->width_length +
                  film->height_length * film->height_length);
    focal_length = diagonal_length / (2.0 * std::tan(0.5 * fov));

    std::cout << "[Camera] focal length: " << focal_length << std::endl;
  }

  // sample ray from camera
  // (i, j): pixel coordinate
  Ray sampleRay(uint32_t i, uint32_t j, Sampler& sampler) const {
    // sample position in pixel with super sampling
    const Real u = film->width_length *
                   (2.0 * (j + sampler.uniformReal()) - film->width) /
                   film->height;
    const Real v = film->height_length *
                   (2.0 * (i + sampler.uniformReal()) - film->height) /
                   film->height;
    const Vec3 p_film = origin + u * right + v * up;

    // sample ray
    const Vec3 p_pinhole = origin + focal_length * forward;
    return Ray(origin, normalize(p_pinhole - p_film));
  }
};

//////////////////////////////////////////

class Material {
  // it represents BRDF
  // computations are done in local coordinate(surface normal is y-axis)
 public:
  // sample direction propotional to BRDF
  // wo: reversed ray direction
  // direction: sampled direction
  // pdf_solid: pdf of direction
  virtual Vec3 sampleBRDF(const Vec3& wo, Sampler& sampler, Vec3& direction,
                          Real& pdf_solid) const = 0;
};

// Material utils

Real absCosTheta(const Vec3& w) { return std::abs(w.y); }

// reflect v with normal n
Vec3 reflect(const Vec3& v, const Vec3& n) { return -v + 2 * dot(v, n) * n; }

class Diffuse : public Material {
  // Lambertian BRDF
 public:
  const Vec3 kd;  // reflectance

  Diffuse(const Vec3& _kd) : kd(_kd) {}

  Vec3 sampleBRDF(const Vec3& wo, Sampler& sampler, Vec3& direction,
                  Real& pdf_solid) const override {
    // sample direction
    direction = sampleCosineHemisphere(sampler.uniformReal(),
                                       sampler.uniformReal(), pdf_solid);

    // compute BRDF
    return INV_PI * kd;
  }
};

class Mirror : public Material {
  // Mirror BRDF
 public:
  const Vec3 ks;

  Mirror(const Vec3& _ks) : ks(_ks) {}

  Vec3 sampleBRDF(const Vec3& wo, Sampler& sampler, Vec3& direction,
                  Real& pdf_solid) const override {
    direction = reflect(wo, Vec3(0, 1, 0));
    pdf_solid = 1;
    return ks / absCosTheta(direction);
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

// prototype declaration of Primitive
class Primitive;

// IntersectInfo
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

class Shape {
  // it represents shape of primitive
 public:
  // compute intersection with ray
  virtual bool intersect(const Ray& ray, IntersectInfo& info) const = 0;
};

class Sphere : public Shape {
 public:
  const Vec3 center;  // center of sphere
  const Real radius;  // radius of sphere

  Sphere(const Vec3& _center, Real _radius)
      : center(_center), radius(_radius) {}

  bool intersect(const Ray& ray, IntersectInfo& info) const override {
    // solve quadratic equation
    const Real b = dot(ray.direction, ray.origin - center);
    const Real c = length2(ray.origin - center) - radius * radius;
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

    const Vec3 r = info.hitPos - center;
    info.hitNormal = normalize(r);

    info.dpdu = normalize(Vec3(-r.z, 0, r.x));

    // compute local coordinate(phi, theta) of hit position
    Real phi = std::atan2(r.z, r.x);
    if (phi < 0) phi += PI2;
    const Real theta =
        std::acos(std::clamp(r.y / radius, Real(-1.0), Real(1.0)));

    info.dpdv = normalize(Vec3(std::cos(phi) * r.y, -radius * std ::sin(theta),
                               std::sin(phi) * r.y));

    return true;
  }
};

class Plane : public Shape {
 public:
  Vec3 leftCornerPoint;
  Vec3 right;
  Vec3 up;
  Vec3 normal;

  Vec3 center;
  Vec3 rightDir;
  Real rightLength;
  Vec3 upDir;
  Real upLength;

  Plane(const Vec3& _leftCornerPoint, const Vec3& _up, const Vec3& _right)
      : leftCornerPoint(_leftCornerPoint), right(_right), up(_up) {
    normal = normalize(cross(right, up));
    center = leftCornerPoint + 0.5 * right + 0.5 * up;
    rightDir = normalize(right);
    rightLength = length(right);
    upDir = normalize(up);
    upLength = length(up);
  };

  bool intersect(const Ray& ray, IntersectInfo& res) const override {
    const Real t =
        -dot(ray.origin - center, normal) / dot(ray.direction, normal);
    if (t < ray.tmin || t > ray.tmax) return false;

    const Vec3 hitPos = ray(t);
    const Real dx = dot(hitPos - leftCornerPoint, rightDir);
    const Real dy = dot(hitPos - leftCornerPoint, upDir);
    if (dx < 0 || dx > rightLength || dy < 0 || dy > upLength) return false;

    res.t = t;
    res.hitPos = hitPos;
    res.hitNormal = dot(-ray.direction, normal) > 0 ? normal : -normal;
    res.dpdu = rightDir;
    res.dpdv = upDir;
    return true;
  };
};

//////////////////////////////////////////

class Primitive {
 public:
  std::shared_ptr<Shape> shape;        // shape of primitive
  std::shared_ptr<Material> material;  // material of primitive
  std::shared_ptr<Light> light;        // area light of primitive

  Primitive(const std::shared_ptr<Shape>& _shape,
            const std::shared_ptr<Material>& _material,
            const std::shared_ptr<Light>& _light)
      : shape(_shape), material(_material), light(_light) {}

  bool hasLight() const { return light != nullptr; }

  // intersect ray with primitive
  bool intersect(const Ray& ray, IntersectInfo& info) const {
    return shape->intersect(ray, info);
  }

  // sample direction propotional to BRDF
  Vec3 sampleBRDF(const Ray& ray, const IntersectInfo& info, Sampler& sampler,
                  Vec3& direction, Real& pdf_solid) const {
    // convert direction vector from world to local
    const Vec3 wo =
        worldToLocal(-ray.direction, info.dpdu, info.hitNormal, info.dpdv);

    Vec3 direction_local;
    const Vec3 BRDF =
        material->sampleBRDF(wo, sampler, direction_local, pdf_solid);

    // convert direction vector from local to world
    direction = normalize(
        localToWorld(direction_local, info.dpdu, info.hitNormal, info.dpdv));

    return BRDF;
  }
};

//////////////////////////////////////////

class Intersector {
  // it computes ray's intersection with primitives
 public:
  const std::vector<std::shared_ptr<Primitive>> prims;  // primitives

  Intersector(const std::vector<std::shared_ptr<Primitive>>& _prims)
      : prims(_prims) {}

  // find closest intersection
  virtual bool intersect(const Ray& ray, IntersectInfo& info) const = 0;
};

class LinearIntersector : public Intersector {
 public:
  LinearIntersector(const std::vector<std::shared_ptr<Primitive>>& _prims)
      : Intersector(_prims) {}

  // find closest intersection by linear search
  bool intersect(const Ray& ray, IntersectInfo& info) const override {
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

class Sky {
 public:
  const Vec3 le;

  Sky(const Vec3& _le) : le(_le) {}

  // compute radiance from sky
  Vec3 Le(const Ray& ray) const { return le; }
};

//////////////////////////////////////////

class Scene {
 public:
  const std::shared_ptr<Camera> camera;
  const std::vector<std::shared_ptr<Primitive>> prims;
  const std::shared_ptr<Sky> sky;

  std::shared_ptr<Intersector> intersector;

  Scene(const std::shared_ptr<Camera>& _camera,
        const std::vector<std ::shared_ptr<Primitive>>& _prims,
        const std ::shared_ptr<Sky>& _sky)
      : camera(_camera), prims(_prims), sky(_sky) {
    // setup intersector
    intersector = std::make_shared<LinearIntersector>(prims);
  }

  // compute intersection of given ray with scene
  bool intersect(const Ray& ray, IntersectInfo& info) const {
    return intersector->intersect(ray, info);
  }
};

//////////////////////////////////////////

class Integrator {
  // integrator computes given ray's radiance.
  // path tracing is implemented.
 public:
  const uint64_t maxDepth = 100;            // maximum number of reflection
  const Real russian_roulette_prob = 0.99;  // probability of russian roulette

  Integrator() {}

  // compute given ray's radiance
  Vec3 radiance(const Ray& ray_in, const Scene& scene, Sampler& sampler) const {
    Ray ray = ray_in;
    Vec3 radiance;
    Vec3 throughput(1);

    for (uint64_t depth = 0; depth < maxDepth; depth++) {
      // russian roulette
      if (sampler.uniformReal() >= russian_roulette_prob) break;
      throughput /= russian_roulette_prob;

      // compute intersection with scene
      IntersectInfo info;
      if (scene.intersect(ray, info)) {
        const auto& prim = info.hitPrimitive;

        // Le
        if (prim->hasLight()) {
          radiance += throughput * prim->light->Le();
        }

        // BRDF sampling
        Vec3 next_direction;
        Real pdf_solid;
        const Vec3 BRDF =
            prim->sampleBRDF(ray, info, sampler, next_direction, pdf_solid);

        // cosine term
        const Real cos = std::abs(dot(next_direction, info.hitNormal));

        // update throughput
        throughput *= BRDF * cos / pdf_solid;

        // update ray
        ray = Ray(info.hitPos, next_direction);

      } else {
        radiance += throughput * scene.sky->Le(ray);
        break;
      }
    }

    return radiance;
  }
};

//////////////////////////////////////////

// print utility

// percentage string
std::string percentage(Real x, Real max) {
  return std::to_string(x / max * 100.0) + "%";
}
// progressbar string
std::string progressbar(Real x, Real max) {
  const int max_count = 40;
  int cur_count = (int)(x / max * max_count);
  std::string str;
  str += "[";
  for (int i = 0; i < cur_count; i++) str += "#";
  for (int i = 0; i < (max_count - cur_count - 1); i++) str += " ";
  str += "]";
  return str;
}

//////////////////////////////////////////

class Renderer {
 public:
  const Scene scene;
  const Integrator integrator;
  Sampler sampler;

  Renderer(const Scene& _scene, const Integrator& _integrator,
           const Sampler& _sampler)
      : scene(_scene), integrator(_integrator), sampler(_sampler) {}

  // n_samples: number of samples
  void render(uint64_t n_samples) {
    const auto start_time = std::chrono::system_clock::now();
    for (uint64_t k = 0; k < n_samples; ++k) {
#pragma omp parallel for schedule(dynamic, 1)
      for (uint32_t i = 0; i < scene.camera->film->height; ++i) {
        for (uint32_t j = 0; j < scene.camera->film->width; ++j) {
          // sample ray
          const Ray ray = scene.camera->sampleRay(i, j, sampler);

          // compute radiance
          const Vec3 radiance = integrator.radiance(ray, scene, sampler);

          // add radiance on pixel
          scene.camera->film->addPixel(i, j, radiance);

          if (omp_get_thread_num() == 0) {
            const Real index =
                j + i * scene.camera->film->width +
                k * scene.camera->film->width * scene.camera->film->height;
            const Real max = scene.camera->film->width *
                             scene.camera->film->height * n_samples;
            std::cout << progressbar(index, max) << " "
                      << percentage(index, max) << "%"
                      << "\r" << std::flush;
          }
        }
      }
    }

    // show rendering time
    const auto end_time = std::chrono::system_clock::now();
    const auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_time - start_time)
                          .count();
    std::cout << "Rendering finished in " << msec << "ms" << std::endl;

    // divide by n_samples
    scene.camera->film->divide(n_samples);

    // write ppm
    scene.camera->film->writePPM("output.ppm");
  }
};

//////////////////////////////////////////

// Scenes

Scene testScene(const std::shared_ptr<Film>& film) {
  // setup camera
  const auto camera =
      std::make_shared<Camera>(Vec3(0, 1, 3), Vec3(0, 0, -1), film, PI / 2.0);

  // setup primitives
  std::vector<std::shared_ptr<Primitive>> prims;
  prims.push_back(std::make_shared<Primitive>(
      std::make_shared<Plane>(Vec3(-2, 0, -2), Vec3(0, 0, 4), Vec3(4, 0, 0)),
      std::make_shared<Diffuse>(Vec3(0.8)), std::make_shared<Light>(Vec3(0))));
  prims.push_back(std::make_shared<Primitive>(
      std::make_shared<Sphere>(Vec3(0, 1, 0), 1),
      std::make_shared<Diffuse>(Vec3(0.2, 0.2, 0.8)),
      std::make_shared<Light>(Vec3(0))));

  // setup sky
  const auto sky = std::make_shared<Sky>(Vec3(1));

  // setup scene
  Scene scene(camera, prims, sky);

  return scene;
}

Scene cornellBoxScene(const std::shared_ptr<Film>& film) {
  // setup camera
  const auto camera = std::make_shared<Camera>(Vec3(278, 273, -900),
                                               Vec3(0, 0, 1), film, PI / 4.0);

  // setup primitives
  std::vector<std::shared_ptr<Primitive>> prims;

  const auto floor = std::make_shared<Plane>(Vec3(0, 0, 0), Vec3(0, 0, 559.2),
                                             Vec3(556, 0, 0));
  const auto right_wall = std::make_shared<Plane>(
      Vec3(0, 0, 0), Vec3(0, 548.8, 0), Vec3(0, 0, 559.2));
  const auto left_wall = std::make_shared<Plane>(
      Vec3(556, 0, 0), Vec3(0, 0, 559.2), Vec3(0, 548.8, 0));
  const auto ceil = std::make_shared<Plane>(Vec3(0, 548.8, 0), Vec3(556, 0, 0),
                                            Vec3(0, 0, 559.2));
  const auto forward_wall = std::make_shared<Plane>(
      Vec3(0, 0, 559.2), Vec3(0, 548.8, 0), Vec3(556, 0, 0));
  const auto shortblock1 = std::make_shared<Plane>(
      Vec3(130, 165, 65), Vec3(-48, 0, 160), Vec3(160, 0, 49));
  const auto shortblock2 = std::make_shared<Plane>(
      Vec3(290, 0, 114), Vec3(0, 165, 0), Vec3(-50, 0, 158));
  const auto shortblock3 = std::make_shared<Plane>(
      Vec3(130, 0, 65), Vec3(0, 165, 0), Vec3(160, 0, 49));
  const auto shortblock4 = std::make_shared<Plane>(
      Vec3(82, 0, 225), Vec3(0, 165, 0), Vec3(48, 0, -160));
  const auto shortblock5 = std::make_shared<Plane>(
      Vec3(240, 0, 272), Vec3(0, 165, 0), Vec3(-158, 0, -47));
  const auto tallblock1 = std::make_shared<Plane>(
      Vec3(423, 330, 247), Vec3(-158, 0, 49), Vec3(49, 0, 159));
  const auto tallblock2 = std::make_shared<Plane>(
      Vec3(423, 0, 247), Vec3(0, 330, 0), Vec3(49, 0, 159));
  const auto tallblock3 = std::make_shared<Plane>(
      Vec3(472, 0, 406), Vec3(0, 330, 0), Vec3(-158, 0, 50));
  const auto tallblock4 = std::make_shared<Plane>(
      Vec3(314, 0, 456), Vec3(0, 330, 0), Vec3(-49, 0, -160));
  const auto tallblock5 = std::make_shared<Plane>(
      Vec3(265, 0, 296), Vec3(0, 330, 0), Vec3(158, 0, -49));
  const auto light = std::make_shared<Plane>(Vec3(343, 548.6, 227),
                                             Vec3(-130, 0, 0), Vec3(0, 0, 105));

  const auto white1 = Vec3(0.8);
  const auto white2 = Vec3(0.99);
  const auto red = Vec3(0.8, 0.05, 0.05);
  const auto green = Vec3(0.05, 0.8, 0.05);

  const auto floor_prim = std::make_shared<Primitive>(
      floor, std::make_shared<Diffuse>(white1), nullptr);
  const auto right_wall_prim = std::make_shared<Primitive>(
      right_wall, std::make_shared<Diffuse>(green), nullptr);
  const auto left_wall_prim = std::make_shared<Primitive>(
      left_wall, std::make_shared<Diffuse>(red), nullptr);
  const auto ceil_prim = std::make_shared<Primitive>(
      ceil, std::make_shared<Diffuse>(white1), nullptr);
  const auto forward_wall_prim = std::make_shared<Primitive>(
      forward_wall, std::make_shared<Diffuse>(white1), nullptr);
  const auto shortblock1_prim = std::make_shared<Primitive>(
      shortblock1, std::make_shared<Diffuse>(white1), nullptr);
  const auto shortblock2_prim = std::make_shared<Primitive>(
      shortblock2, std::make_shared<Diffuse>(white1), nullptr);
  const auto shortblock3_prim = std::make_shared<Primitive>(
      shortblock3, std::make_shared<Diffuse>(white1), nullptr);
  const auto shortblock4_prim = std::make_shared<Primitive>(
      shortblock4, std::make_shared<Diffuse>(white1), nullptr);
  const auto shortblock5_prim = std::make_shared<Primitive>(
      shortblock5, std::make_shared<Diffuse>(white1), nullptr);
  const auto tallblock1_prim = std::make_shared<Primitive>(
      tallblock1, std::make_shared<Diffuse>(white1), nullptr);
  const auto tallblock2_prim = std::make_shared<Primitive>(
      tallblock2, std::make_shared<Diffuse>(white1), nullptr);
  const auto tallblock3_prim = std::make_shared<Primitive>(
      tallblock3, std::make_shared<Diffuse>(white1), nullptr);
  const auto tallblock4_prim = std::make_shared<Primitive>(
      tallblock4, std::make_shared<Diffuse>(white1), nullptr);
  const auto tallblock5_prim = std::make_shared<Primitive>(
      tallblock5, std::make_shared<Diffuse>(white1), nullptr);
  const auto light_prim = std::make_shared<Primitive>(
      light, std::make_shared<Diffuse>(white1),
      std::make_shared<Light>(0.1 * Vec3(340, 190, 100)));

  prims.push_back(floor_prim);
  prims.push_back(right_wall_prim);
  prims.push_back(left_wall_prim);
  prims.push_back(ceil_prim);
  prims.push_back(forward_wall_prim);
  prims.push_back(shortblock1_prim);
  prims.push_back(shortblock2_prim);
  prims.push_back(shortblock3_prim);
  prims.push_back(shortblock4_prim);
  prims.push_back(shortblock5_prim);
  prims.push_back(tallblock1_prim);
  prims.push_back(tallblock2_prim);
  prims.push_back(tallblock3_prim);
  prims.push_back(tallblock4_prim);
  prims.push_back(tallblock5_prim);
  prims.push_back(light_prim);

  // setup sky
  const auto sky = std::make_shared<Sky>(Vec3(0));

  // setup scene
  Scene scene(camera, prims, sky);

  return scene;
}

//////////////////////////////////////////

int main() {
  // parameters
  const uint32_t width = 512;
  const uint32_t height = 512;
  const uint64_t samples = 100;

  // setup image
  const auto film = std::make_shared<Film>(width, height);

  // setup sampler
  Sampler sampler;

  // setup integrator
  Integrator integrator;

  // setup scene
  Scene scene = cornellBoxScene(film);

  // setup renderer
  Renderer renderer(scene, integrator, sampler);

  // render
  renderer.render(samples);

  return 0;
}