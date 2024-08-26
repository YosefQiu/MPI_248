#pragma once
#include "ggl.h"



class Camera
{
public:
	float3 from;
	float3 to;
	float3 u;
	float3 v;
	float3 dir;
	float3 up;
	float zoom;
	float n, f;
public:
	Camera() = default;
	Camera(const float3& from, const float3& to, const float3& up, float zoom, float n, float f);
public:
	void setRatioUV(float ratio);
};

