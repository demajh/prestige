class Prestige < Formula
  desc "Content-deduplicated key-value store with exact and semantic dedup"
  homepage "https://github.com/demajh/prestige"
  url "https://github.com/demajh/prestige/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "REPLACE_WITH_ACTUAL_SHA256"
  license "Apache-2.0"
  head "https://github.com/demajh/prestige.git", branch: "main"

  depends_on "cmake" => :build
  depends_on "rocksdb"

  # Optional: for semantic dedup
  # depends_on "onnxruntime" => :optional

  def install
    args = %W[
      -DCMAKE_INSTALL_PREFIX=#{prefix}
      -DCMAKE_BUILD_TYPE=Release
      -DPRESTIGE_BUILD_EXAMPLES=OFF
      -DPRESTIGE_BUILD_TOOLS=ON
    ]

    # Uncomment for semantic dedup support
    # if build.with? "onnxruntime"
    #   args << "-DPRESTIGE_ENABLE_SEMANTIC=ON"
    # end

    system "cmake", "-S", ".", "-B", "build", *args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    (testpath/"test.cpp").write <<~EOS
      #include <prestige/store.hpp>
      #include <iostream>

      int main() {
        prestige::Options opt;
        std::unique_ptr<prestige::Store> db;
        auto s = prestige::Store::Open("./test_db", &db, opt);
        if (!s.ok()) {
          std::cerr << "Failed to open: " << s.ToString() << std::endl;
          return 1;
        }
        s = db->Put("key", "value");
        if (!s.ok()) return 1;
        std::string val;
        s = db->Get("key", &val);
        if (!s.ok() || val != "value") return 1;
        std::cout << "Success!" << std::endl;
        return 0;
      }
    EOS

    system ENV.cxx, "-std=c++17", "test.cpp",
           "-I#{include}", "-L#{lib}", "-lprestige_uvs",
           "-I#{Formula["rocksdb"].opt_include}",
           "-L#{Formula["rocksdb"].opt_lib}", "-lrocksdb",
           "-o", "test"
    system "./test"
  end
end
