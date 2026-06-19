#ifndef NN_PREDICTOR_H
#define NN_PREDICTOR_H

/**
 * nn_predictor.h — C++ NN 推理 (LibTorch)
 *
 * 加载 TorchScript 模型，对观测张量做推理
 */

#include <string>
#include <vector>
#include <iostream>

#ifdef USE_TORCH
#include <torch/script.h>
#include <torch/torch.h>

class NNPredictor {
public:
    NNPredictor() : loaded_(false) {}

    /** 加载 TorchScript 模型 */
    bool load(const std::string& model_path) {
        try {
            module_ = torch::jit::load(model_path);
            module_.eval();
            loaded_ = true;
            std::cerr << "[NN] 加载模型: " << model_path << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[NN] 加载失败: " << e.what() << std::endl;
            return false;
        }
    }

    bool is_loaded() const { return loaded_; }

    /**
     * 推理: 输入观测 (7×H×W)，输出 policy (1153) 和 value (-1~1)
     *
     * obs: 7 channels × H × W 的 float 数组
     * policy_out: 输出策略数组 (n_actions)
     * value_out: 输出价值标量
     */
    void predict(const float* obs, int h, int w, int n_actions,
                 float* policy_out, float& value_out) {
        if (!loaded_) {
            std::cerr << "[NN] 模型未加载!" << std::endl;
            return;
        }

        // 零拷贝创建张量（buffer 在 forward 返回前保持有效即可）
        torch::Tensor obs_tensor = torch::from_blob(
            const_cast<float*>(obs), {1, 7, h, w}, torch::kFloat32
        );

        // 推理
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(obs_tensor);
        auto output = module_.forward(inputs);

        // 解析输出: tuple(policy, value)
        auto tuple = output.toTuple();
        at::Tensor policy_tensor = tuple->elements()[0].toTensor();
        at::Tensor value_tensor = tuple->elements()[1].toTensor();

        // 复制 policy
        float* policy_data = policy_tensor.data_ptr<float>();
        std::memcpy(policy_out, policy_data, n_actions * sizeof(float));

        // 复制 value
        value_out = value_tensor.item<float>();
    }

    /** 直接在游戏状态上推理 */
    void predict_on_state(const class GameState& gs, int player,
                          int h, int w, int n_actions,
                          float* policy_out, float& value_out) {
        // 提取观测
        std::vector<float> obs(7 * h * w, 0.0f);
        gs.get_obs(obs.data(), player);
        predict(obs.data(), h, w, n_actions, policy_out, value_out);
    }

private:
    torch::jit::Module module_;
    bool loaded_;
};

#endif // NN_PREDICTOR_H
#endif // USE_TORCH
