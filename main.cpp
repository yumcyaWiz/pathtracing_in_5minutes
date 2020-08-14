// STL
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

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

  Real length() const { return std::sqrt(x * x + y * y + z * z); }
  Real length2() const { return x * x + y * y + z * z; }
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

  IntersectInfo() {}
};

//////////////////////////////////////////

int main() {
  Image img(512, 512);
  img.writePPM("output.ppm");

  return 0;
}