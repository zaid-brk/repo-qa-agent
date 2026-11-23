import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export", // static files only, so Render can serve it as a free Static Site
};

export default nextConfig;
