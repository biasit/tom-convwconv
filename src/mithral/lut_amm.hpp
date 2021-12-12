//
//  lut_amm.hpp
//
// Based on profile_amm.hpp from bolt/cpp/test/profile_amm.hpp

#include <stdio.h>
#include <string>
#include <vector>

#include "mithral.hpp"

class MithralMatmul {

    static constexpr int scan_block_nrows = 32;         // not sure if this is necessary
    static constexpr int ncentroids = 16;
    static constexpr int nsplits_per_codebook = 4;      // when hashing, how many splits to have in the tree
    static constexpr int max_splitvals = 1 << 4;


    public:
        MithralMatmul(int N, int D, int M, int ncodebooks,
                     float lut_work_const):
            N_padded(N % scan_block_nrows == 0 ? N :
                N + (scan_block_nrows - (N % scan_block_nrows))),
            centroids(ncentroids * ncodebooks, D),
            nsplits(ncodebooks * nsplits_per_codebook),
            splitdims(nsplits),
            splitvals(max_splitvals, nsplits),
            encode_scales(nsplits),
            encode_offsets(nsplits),
            nnz_per_centroid(lut_work_const > 0 ?
                lut_work_const * D / ncodebooks : D),
            idxs(ncodebooks, nnz_per_centroid),
            amm(N_padded, D, M, ncodebooks, centroids.data(),
                splitdims.data(), splitvals.data(),
                encode_scales.data(), encode_offsets.data(),
                idxs.data(), nnz_per_centroid),
            X(N_padded, D),
            Q(D, M)
        {
            centroids.setRandom();
            splitdims.setRandom();
            for (int i = 0; i < splitdims.size(); i++) {
                splitdims(i) = splitdims(i) % D;
            }
            splitvals.setRandom();
            encode_scales.setRandom();
            encode_offsets.setRandom();

            // randomly initialize idxs, ensuring all are unique and < D
            idxs.setRandom();
            int all_idxs[D];
            for (int i = 0; i < D; i++) {
                all_idxs[i] = i;
            }
            std::random_device rd;
            std::mt19937 g(rd());  // why can't shuffle just create its own...
            for (int c = 0; c < ncodebooks; c++) {  // random sequential idxs
                std::shuffle(all_idxs, all_idxs + D, g);
                std::sort(all_idxs, all_idxs + nnz_per_centroid);
                for (int j = 0; j < nnz_per_centroid; j++) {
                    idxs(c, j) = all_idxs[j];
                }
            }

            X.setRandom();
            Q.setRandom();
        }

        void run_matmul(bool create_lut=true) {
            encode();
            if (create_lut) {
                lut();
            }
            scan();
        }

        const Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& output() const { return amm.out_mat; }

        // stuff we pass into the amm object (would be learned during training)
        int N_padded;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> centroids;
        int nsplits;
        Eigen::Matrix<uint32_t, 1, Eigen::Dynamic, Eigen::RowMajor> splitdims;
        Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> splitvals;
        Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> encode_scales;
        Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> encode_offsets;
        int nnz_per_centroid;
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> idxs;

        // random data
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> X;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Q;

    private:
        // amm object
        mithral_amm<float> amm;

        void encode() { amm.encode(X.data()); }
        void lut() { amm.lut(Q.data()); }
        void scan() { amm.scan(); }
};
