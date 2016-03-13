// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: inception_inference.proto

#ifndef PROTOBUF_inception_5finference_2eproto__INCLUDED
#define PROTOBUF_inception_5finference_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3000000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace tensorflow {
namespace serving {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_inception_5finference_2eproto();
void protobuf_AssignDesc_inception_5finference_2eproto();
void protobuf_ShutdownFile_inception_5finference_2eproto();

class InceptionRequest;
class InceptionResponse;

// ===================================================================

class InceptionRequest : public ::google::protobuf::Message {
 public:
  InceptionRequest();
  virtual ~InceptionRequest();

  InceptionRequest(const InceptionRequest& from);

  inline InceptionRequest& operator=(const InceptionRequest& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const InceptionRequest& default_instance();

  void Swap(InceptionRequest* other);

  // implements Message ----------------------------------------------

  inline InceptionRequest* New() const { return New(NULL); }

  InceptionRequest* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const InceptionRequest& from);
  void MergeFrom(const InceptionRequest& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(InceptionRequest* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional bytes image_data = 1;
  void clear_image_data();
  static const int kImageDataFieldNumber = 1;
  const ::std::string& image_data() const;
  void set_image_data(const ::std::string& value);
  void set_image_data(const char* value);
  void set_image_data(const void* value, size_t size);
  ::std::string* mutable_image_data();
  ::std::string* release_image_data();
  void set_allocated_image_data(::std::string* image_data);

  // @@protoc_insertion_point(class_scope:tensorflow.serving.InceptionRequest)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  bool _is_default_instance_;
  ::google::protobuf::internal::ArenaStringPtr image_data_;
  mutable int _cached_size_;
  friend void  protobuf_AddDesc_inception_5finference_2eproto();
  friend void protobuf_AssignDesc_inception_5finference_2eproto();
  friend void protobuf_ShutdownFile_inception_5finference_2eproto();

  void InitAsDefaultInstance();
  static InceptionRequest* default_instance_;
};
// -------------------------------------------------------------------

class InceptionResponse : public ::google::protobuf::Message {
 public:
  InceptionResponse();
  virtual ~InceptionResponse();

  InceptionResponse(const InceptionResponse& from);

  inline InceptionResponse& operator=(const InceptionResponse& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const InceptionResponse& default_instance();

  void Swap(InceptionResponse* other);

  // implements Message ----------------------------------------------

  inline InceptionResponse* New() const { return New(NULL); }

  InceptionResponse* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const InceptionResponse& from);
  void MergeFrom(const InceptionResponse& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const;
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(InceptionResponse* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated float value = 1 [packed = true];
  int value_size() const;
  void clear_value();
  static const int kValueFieldNumber = 1;
  float value(int index) const;
  void set_value(int index, float value);
  void add_value(float value);
  const ::google::protobuf::RepeatedField< float >&
      value() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_value();

  // @@protoc_insertion_point(class_scope:tensorflow.serving.InceptionResponse)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  bool _is_default_instance_;
  ::google::protobuf::RepeatedField< float > value_;
  mutable int _value_cached_byte_size_;
  mutable int _cached_size_;
  friend void  protobuf_AddDesc_inception_5finference_2eproto();
  friend void protobuf_AssignDesc_inception_5finference_2eproto();
  friend void protobuf_ShutdownFile_inception_5finference_2eproto();

  void InitAsDefaultInstance();
  static InceptionResponse* default_instance_;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// InceptionRequest

// optional bytes image_data = 1;
inline void InceptionRequest::clear_image_data() {
  image_data_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline const ::std::string& InceptionRequest::image_data() const {
  // @@protoc_insertion_point(field_get:tensorflow.serving.InceptionRequest.image_data)
  return image_data_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void InceptionRequest::set_image_data(const ::std::string& value) {
  
  image_data_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:tensorflow.serving.InceptionRequest.image_data)
}
inline void InceptionRequest::set_image_data(const char* value) {
  
  image_data_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:tensorflow.serving.InceptionRequest.image_data)
}
inline void InceptionRequest::set_image_data(const void* value, size_t size) {
  
  image_data_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:tensorflow.serving.InceptionRequest.image_data)
}
inline ::std::string* InceptionRequest::mutable_image_data() {
  
  // @@protoc_insertion_point(field_mutable:tensorflow.serving.InceptionRequest.image_data)
  return image_data_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* InceptionRequest::release_image_data() {
  
  return image_data_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void InceptionRequest::set_allocated_image_data(::std::string* image_data) {
  if (image_data != NULL) {
    
  } else {
    
  }
  image_data_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), image_data);
  // @@protoc_insertion_point(field_set_allocated:tensorflow.serving.InceptionRequest.image_data)
}

// -------------------------------------------------------------------

// InceptionResponse

// repeated float value = 1 [packed = true];
inline int InceptionResponse::value_size() const {
  return value_.size();
}
inline void InceptionResponse::clear_value() {
  value_.Clear();
}
inline float InceptionResponse::value(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.serving.InceptionResponse.value)
  return value_.Get(index);
}
inline void InceptionResponse::set_value(int index, float value) {
  value_.Set(index, value);
  // @@protoc_insertion_point(field_set:tensorflow.serving.InceptionResponse.value)
}
inline void InceptionResponse::add_value(float value) {
  value_.Add(value);
  // @@protoc_insertion_point(field_add:tensorflow.serving.InceptionResponse.value)
}
inline const ::google::protobuf::RepeatedField< float >&
InceptionResponse::value() const {
  // @@protoc_insertion_point(field_list:tensorflow.serving.InceptionResponse.value)
  return value_;
}
inline ::google::protobuf::RepeatedField< float >*
InceptionResponse::mutable_value() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.serving.InceptionResponse.value)
  return &value_;
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace serving
}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_inception_5finference_2eproto__INCLUDED
