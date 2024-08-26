#include "Camera.h"



Camera::Camera(const float3& from, const float3& to, const float3& up, float zoom, float n, float f)
{
	this->from = from;
	this->to = to;
	this->zoom = zoom;
	this->n = n;
	this->f = f;
	this->up = up;

	this->dir = normalize(to - from);
	this->u = normalize(cross(up, dir));
	this->v = normalize(cross(dir, u));
}

void Camera::setRatioUV(float ratio)
{
	v = normalize(v);
	v = v * length(u) / ratio;
}
