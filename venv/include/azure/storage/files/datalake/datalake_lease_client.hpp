// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "azure/storage/files/datalake/datalake_file_system_client.hpp"
#include "azure/storage/files/datalake/datalake_path_client.hpp"

#include <azure/storage/blobs/blob_lease_client.hpp>

#include <chrono>

namespace Azure { namespace Storage { namespace Files { namespace DataLake {

  /**
   * @brief DataLakeLeaseClient allows you to manipulate Azure Storage leases on filesystems and
   * paths.
   */
  class DataLakeLeaseClient final {
  public:
    /**
     * @brief Initializes a new instance of the DataLakeLeaseClient.
     *
     * @param pathClient A DataLakePathClient representing the datalake path being leased.
     * @param leaseId A lease ID. This is not required for break operation.
     */
    explicit DataLakeLeaseClient(DataLakePathClient pathClient, std::string leaseId)
        : m_blobLeaseClient(std::move(pathClient.m_blobClient), std::move(leaseId))
    {
    }

    /**
     * @brief Initializes a new instance of the DataLakeLeaseClient.
     *
     * @param fileSystemClient A DataLakeFileSystemClient representing the filesystem being leased.
     * @param leaseId A lease ID. This is not required for break operation.
     */
    explicit DataLakeLeaseClient(DataLakeFileSystemClient fileSystemClient, std::string leaseId)
        : m_blobLeaseClient(std::move(fileSystemClient.m_blobContainerClient), std::move(leaseId))
    {
    }

    /**
     * @brief Gets a unique lease ID.
     *
     * @return A unique lease ID.
     */
    static std::string CreateUniqueLeaseId()
    {
      return Blobs::BlobLeaseClient::CreateUniqueLeaseId();
    };

    /**
     * @brief A value representing infinite lease duration.
     */
    AZ_STORAGE_FILES_DATALAKE_DLLEXPORT const static std::chrono::seconds InfiniteLeaseDuration;

    /**
     * @brief Get lease ID of this lease client.
     *
     * @return Lease ID of this lease client.
     */
    std::string GetLeaseId() { return m_blobLeaseClient.GetLeaseId(); }

    /**
     * @brief Acquires a lease on the datalake path or datalake path container.
     *
     * @param duration Specifies the duration of
     * the lease, in seconds, or InfiniteLeaseDuration for a lease that never
     * expires. A non-infinite lease can be between 15 and 60 seconds. A lease duration cannot be
     * changed using renew or change.
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return An AcquireLeaseResult describing the lease.
     */
    Azure::Response<Models::AcquireLeaseResult> Acquire(
        std::chrono::seconds duration,
        const AcquireLeaseOptions& options = AcquireLeaseOptions(),
        const Azure::Core::Context& context = Azure::Core::Context())
    {
      return m_blobLeaseClient.Acquire(duration, options, context);
    }

    /**
     * @brief Renews the datalake path or datalake path container's previously-acquired lease.
     *
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return A RenewLeaseResult describing the lease.
     */
    Azure::Response<Models::RenewLeaseResult> Renew(
        const RenewLeaseOptions& options = RenewLeaseOptions(),
        const Azure::Core::Context& context = Azure::Core::Context())
    {
      return m_blobLeaseClient.Renew(options, context);
    }

    /**
     * @brief Releases the datalake path or datalake path container's previously-acquired lease.
     *
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return A ReleaseLeaseResult describing the updated container or blob.
     */
    Azure::Response<Models::ReleaseLeaseResult> Release(
        const ReleaseLeaseOptions& options = ReleaseLeaseOptions(),
        const Azure::Core::Context& context = Azure::Core::Context())
    {
      return m_blobLeaseClient.Release(options, context);
    }

    /**
     * @brief Changes the lease of an active lease.
     *
     * @param proposedLeaseId Proposed lease ID, in a GUID string format.
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return A ChangeLeaseResult describing the changed lease.
     * @remarks The current DataLakeLeaseClient becomes invalid if this operation succeeds.
     */
    Azure::Response<Models::ChangeLeaseResult> Change(
        const std::string& proposedLeaseId,
        const ChangeLeaseOptions& options = ChangeLeaseOptions(),
        const Azure::Core::Context& context = Azure::Core::Context())
    {
      return m_blobLeaseClient.Change(proposedLeaseId, options, context);
    }

    /**
     * @brief Breaks the previously-acquired lease.
     *
     * @param options Optional parameters to execute this function.
     * @param context Context for cancelling long running operations.
     * @return A BreakLeaseResult describing the broken lease.
     */
    Azure::Response<Models::BreakLeaseResult> Break(
        const BreakLeaseOptions& options = BreakLeaseOptions(),
        const Azure::Core::Context& context = Azure::Core::Context())
    {
      return m_blobLeaseClient.Break(options, context);
    }

  private:
    Blobs::BlobLeaseClient m_blobLeaseClient;
  };

}}}} // namespace Azure::Storage::Files::DataLake
