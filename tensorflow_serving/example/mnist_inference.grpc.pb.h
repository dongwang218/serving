/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// TODO(b/26692266): Automatically generate gRPC proto files.
// For now follow instructions in the bug.

// Generated by the gRPC protobuf plugin.
// If you make any local change, they will be lost.
// source: mnist_inference.proto
#ifndef GRPC_mnist_5finference_2eproto__INCLUDED
#define GRPC_mnist_5finference_2eproto__INCLUDED

#include "mnist_inference.pb.h"

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>

namespace grpc {
class CompletionQueue;
class RpcService;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc

namespace tensorflow {
namespace serving {

class MnistService GRPC_FINAL {
 public:
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status Classify(::grpc::ClientContext* context, const ::tensorflow::serving::MnistRequest& request, ::tensorflow::serving::MnistResponse* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::MnistResponse>> AsyncClassify(::grpc::ClientContext* context, const ::tensorflow::serving::MnistRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::MnistResponse>>(AsyncClassifyRaw(context, request, cq));
    }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::tensorflow::serving::MnistResponse>* AsyncClassifyRaw(::grpc::ClientContext* context, const ::tensorflow::serving::MnistRequest& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub GRPC_FINAL : public StubInterface {
   public:
    Stub(const std::shared_ptr<::grpc::ChannelInterface>& channel);
    ::grpc::Status Classify(::grpc::ClientContext* context, const ::tensorflow::serving::MnistRequest& request, ::tensorflow::serving::MnistResponse* response) GRPC_OVERRIDE;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::MnistResponse>> AsyncClassify(::grpc::ClientContext* context, const ::tensorflow::serving::MnistRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::MnistResponse>>(AsyncClassifyRaw(context, request, cq));
    }

   private:
    std::shared_ptr<::grpc::ChannelInterface> channel_;
    ::grpc::ClientAsyncResponseReader< ::tensorflow::serving::MnistResponse>* AsyncClassifyRaw(::grpc::ClientContext* context, const ::tensorflow::serving::MnistRequest& request, ::grpc::CompletionQueue* cq) GRPC_OVERRIDE;
    const ::grpc::RpcMethod rpcmethod_Classify_;
  };
  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr<::grpc::ChannelInterface>& channel,
      const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status Classify(::grpc::ServerContext* context, const ::tensorflow::serving::MnistRequest* request, ::tensorflow::serving::MnistResponse* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_Classify : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(Service* service) {}

   public:
    WithAsyncMethod_Classify() { ::grpc::Service::MarkMethodAsync(0); }
    ~WithAsyncMethod_Classify() GRPC_OVERRIDE {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Classify(::grpc::ServerContext* context,
                            const ::tensorflow::serving::MnistRequest* request,
                            ::tensorflow::serving::MnistResponse* response)
        GRPC_FINAL GRPC_OVERRIDE {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestClassify(
        ::grpc::ServerContext* context,
        ::tensorflow::serving::MnistRequest* request,
        ::grpc::ServerAsyncResponseWriter<::tensorflow::serving::MnistResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq, void* tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response,
                                         new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_Classify<Service> AsyncService;
  template <class BaseClass>
  class WithGenericMethod_Classify : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(Service* service) {}

   public:
    WithGenericMethod_Classify() { ::grpc::Service::MarkMethodGeneric(0); }
    ~WithGenericMethod_Classify() GRPC_OVERRIDE {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Classify(::grpc::ServerContext* context,
                            const ::tensorflow::serving::MnistRequest* request,
                            ::tensorflow::serving::MnistResponse* response)
        GRPC_FINAL GRPC_OVERRIDE {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
};

}  // namespace serving
}  // namespace tensorflow


#endif  // GRPC_mnist_5finference_2eproto__INCLUDED
