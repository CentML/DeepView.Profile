{
  description = "Application packaged using poetry2nix";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    {
      # Nixpkgs overlay providing the application
      overlay = nixpkgs.lib.composeManyExtensions [
        poetry2nix.overlay
        (final: prev: {
          # The application
          skyline = prev.poetry2nix.mkPoetryApplication {
            projectDir = ./.;
          };
        })
      ];
    } // (flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ self.overlay ];
        };
      in
      {
        apps = {
          skyline = pkgs.myapp;
        };

        defaultApp = pkgs.skyline;

        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            (python39.withPackages (ps: with ps; [ poetry ]))
          ];
          packages = with pkgs; [
           autoconf
           automake
           binutils
           boost
           clang
           cmake
           cudaPackages.cuda_cupti
           cudatoolkit
           curl
           freeglut
           git
           gitRepo
           gnome.gnome-keyring # needed for `typed-extensions` to compile`
           gnumake
           gnupg
           gperf
           libGL
           libGLU
           linuxPackages.nvidia_x11
           m4
           ncurses5
           ninja
           openvscode-server
           poetry
           procps
           python39
           python39Packages.ipython
           python39Packages.jupyter
           python39Packages.pip
           python39Packages.setuptools
           stdenv.cc
           unzip
           util-linux
           xorg.libX11
           xorg.libXext
           xorg.libXi
           xorg.libXmu
           xorg.libXrandr
           xorg.libXv
           zlib 
         ];
         shellHook = with pkgs; ''
            export CUDA_PATH=${cudatoolkit}
            export LD_LIBRARY_PATH="${linuxPackages.nvidia_x11}/lib:${ncurses5}/lib:${stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib/"
            export EXTRA_LDFLAGS="-L/lib -L${linuxPackages.nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-I/usr/include"
         '';          
        };
     }
    )
  );
}

