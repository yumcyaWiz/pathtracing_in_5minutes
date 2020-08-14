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

// 3-dimensional vector
class Vec3 {
 public:
  Real x;
  Real y;
  Real z;

  Vec3() { x = y = z = 0; }
  Vec3(Real _x) { x = y = z = _x; }
  Vec3(Real _x, Real _y, Real _z) : x(_x), y(_y), z(_z) {}
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

std::ostream& operator<<(std::ostream& stream, const Vec3& v) {
  stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  return stream;
}

//////////////////////////////////////////

// Ray
class Ray {
 public:
  Vec3 origin;
  Vec3 direction;

  static constexpr Real tmin = std::numeric_limits<Real>::epsilon();
  static constexpr Real tmax = std::numeric_limits<Real>::max();

  Ray() {}
  Ray(const Vec3& _origin, const Vec3& _direction)
      : origin(_origin), direction(_direction) {}

  Vec3 operator()(Real t) const { return origin + t * direction; }
};

//////////////////////////////////////////

// Image
class Image {
 public:
  uint32_t width;   // width of image in [px]
  uint32_t height;  // height of image in [px]
  Vec3* pixels;     // an array contains RGB at each pixel, row-major.

  Image(uint32_t _width, uint32_t _height) : width(_width), height(_height) {
    pixels = new Vec3[width * height];
  }
  ~Image() { delete[] pixels; }

  // getter and setter
  Vec3 getPixel(uint32_t i, uint32_t j) const { return pixels[j + width * i]; }
  void setPixel(uint32_t i, uint32_t j, const Vec3& rgb) {
    pixels[j + width * i] = rgb;
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
        const Vec3& rgb = getPixel(i, j);
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

// IntersectInfo
struct IntersectInfo {
  Real t;          // hit distance
  Vec3 hitPos;     // hit potision
  Vec3 hitNormal;  // surface normal at hit position

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

    return true;
  }
};

//////////////////////////////////////////

// Intersector
class Intersector {
 public:
  std::vector<std::shared_ptr<Sphere>> prims;  // primitives

  Intersector() {}
  Intersector(const std::vector<std::shared_ptr<Sphere>>& _prims) {}

  // find closest intersection linearly
  bool intersect(const Ray& ray, IntersectInfo& info) const {
    bool hit = false;
    Real t = ray.tmax;
    for (const auto& prim : prims) {
      IntersectInfo temp_info;
      if (prim->intersect(ray, temp_info)) {
        if (temp_info.t < t) {
          hit = true;
          info = temp_info;
          t = temp_info.t;
        }
      }
    }
    return hit;
  }
};

//////////////////////////////////////////

// Camera
class Camera {
 public:
  Vec3 origin;   // origin of camera
  Vec3 forward;  // forward direction of camera
  Vec3 right;    // right direction of camera
  Vec3 up;       // up direction of camera

  Camera(const Vec3& _origin, const Vec3& _forward)
      : origin(_origin), forward(normalize(_forward)) {
    right = normalize(cross(forward, Vec3(0, 1, 0)));
    up = normalize(cross(right, forward));

    std::cout << "[Camera] origin: " << origin << std::endl;
    std::cout << "[Camera] forward: " << forward << std::endl;
    std::cout << "[Camera] right: " << right << std::endl;
    std::cout << "[Camera] up: " << up << std::endl;
  }

  // sample ray from camera
  bool sampleRay(uint32_t i, uint32_t j) { return true; }
};

int main() {
  // parameters
  const uint32_t width = 512;
  const uint32_t height = 512;
  const uint32_t samples = 100;

  // setup image
  Image image(width, height);

  // setup scene
  std::vector<std::shared_ptr<Sphere>> prims;
  prims.push_back(std::make_shared<Sphere>(Vec3(0, -10000, 0), 10000));
  prims.push_back(std::make_shared<Sphere>(Vec3(0, 1, 0), 1));

  // setup intersector
  Intersector intersector(prims);

  // path tracing
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      for (int k = 0; k < samples; ++k) {
      }
    }
  }

  // write ppm
  image.writePPM("output.ppm");

  return 0;
}