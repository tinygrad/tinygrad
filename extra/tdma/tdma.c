#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/pci.h>
#include <linux/pci-p2pdma.h>
#include <linux/dma-buf.h>
#include "tdma.h"

// To see pr_debug logs:
// echo 'module tdma +p' | sudo tee /sys/kernel/debug/dynamic_debug/control

struct tdma_priv {
  struct pci_dev *pdev;
  int bar;
  resource_size_t bar_start;
  resource_size_t bar_size;
  int usgl_size;
  struct tdma_ioctl_usge* usgl;
};

static int tdma_attach(struct dma_buf *dmabuf, struct dma_buf_attachment *attach) {
  struct tdma_priv *priv = dmabuf->priv;
  pr_debug("tdma: %s is attaching to %s", dev_name(attach->dev), dev_name(&priv->pdev->dev));

  // Negative distance = can't reach
  if (pci_p2pdma_distance(priv->pdev, attach->dev, true) < 0) {
    pr_err("tdma: %s can't reach %s", dev_name(attach->dev), dev_name(&priv->pdev->dev));
    return -EINVAL;
  }

  return 0;
}

static void tdma_detach(struct dma_buf *dmabuf, struct dma_buf_attachment *attach) {
  struct tdma_priv *priv = dmabuf->priv;
  pr_debug("tdma: %s is detached from %s", dev_name(attach->dev), dev_name(&priv->pdev->dev));
}

static struct sg_table *tdma_map(struct dma_buf_attachment *attach, enum dma_data_direction dir) {
  struct dma_buf* dmabuf = attach->dmabuf;
  struct tdma_priv *priv = dmabuf->priv;
  pr_debug("tdma: %s is mapping %s", dev_name(attach->dev), dev_name(&priv->pdev->dev));

  struct sg_table* sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
  if (!sgt) {
    pr_err("tdma: failed to allocate memory for sg_table");
    return ERR_PTR(-ENOMEM);
  }

  if (sg_alloc_table(sgt, priv->usgl_size, GFP_KERNEL)) {
    pr_err("tdma: failed to allocate sg_table");
    kfree(sgt);
    return ERR_PTR(-ENOMEM);
  }

  struct scatterlist* sg;
  unsigned int i;
  for_each_sgtable_sg(sgt, sg, i) {
    struct tdma_ioctl_usge usge = priv->usgl[i];

    dma_addr_t dma_addr = dma_map_resource(attach->dev, priv->bar_start + usge.offset, usge.size, dir, DMA_ATTR_SKIP_CPU_SYNC);

    if (dma_mapping_error(attach->dev, dma_addr)) {
      pr_err("tdma: dma mapping error");
      struct scatterlist* unmap_sg;
      unsigned int unmap_i;
      for_each_sgtable_sg(sgt, unmap_sg, unmap_i) {
        if (unmap_i >= i) continue;
        pr_debug("tdma: destroying sg[%d] dma=0x%llx size=0x%x", unmap_i, unmap_sg->dma_address, unmap_sg->dma_length);
        dma_unmap_resource(attach->dev, unmap_sg->dma_address, unmap_sg->dma_length, dir, DMA_ATTR_SKIP_CPU_SYNC);
      }
      sg_free_table(sgt);
      kfree(sgt);
      return ERR_PTR(-EIO);
    }

    sg_set_page(sg, NULL, usge.size, 0);
    sg_dma_address(sg) = dma_addr;
    sg_dma_len(sg) = usge.size;
  }

  return sgt;
}

static void tdma_unmap(struct dma_buf_attachment *attach, struct sg_table *sgt, enum dma_data_direction dir) {
  struct dma_buf* dmabuf = attach->dmabuf;
  struct tdma_priv *priv = dmabuf->priv;
  pr_debug("tdma: %s is unmapping %s", dev_name(attach->dev), dev_name(&priv->pdev->dev));
  struct scatterlist* sg;
  unsigned int i;
  for_each_sgtable_sg(sgt, sg, i) {
    pr_debug("tdma: destroying sg[%d] dma=0x%llx size=0x%x", i, sg->dma_address, sg->dma_length);
    dma_unmap_resource(attach->dev, sg->dma_address, sg->dma_length, dir, DMA_ATTR_SKIP_CPU_SYNC);
  }
  sg_free_table(sgt);
  kfree(sgt);
}

static void tdma_release(struct dma_buf *dmabuf) {
  struct tdma_priv *priv = dmabuf->priv;
  pr_debug("tdma: released dmabuf for %s", dev_name(&priv->pdev->dev));
  pci_dev_put(priv->pdev);
  kfree(priv->usgl);
  kfree(priv);
}

static const struct dma_buf_ops tdma_ops = {
  .cache_sgt_mapping = true,
  .attach = tdma_attach,
  .detach = tdma_detach,
  .map_dma_buf = tdma_map,
  .unmap_dma_buf = tdma_unmap,
  .release = tdma_release,
};

static long tdma_ioctl(struct file *filp, unsigned int cmd, unsigned long uarg) {
  if (cmd != TDMA_GET_DMABUF)
    return -ENOTTY;

  // Copy arguments from userspace to kernel
  struct tdma_ioctl arg;
  if (copy_from_user(&arg, (void __user *)uarg, sizeof(arg))) {
    pr_err("tdma: failed to copy ioctl arguments from userspace");
    return -EFAULT;
  }

  struct tdma_ioctl_usge* arg_usgl = kzalloc(sizeof(struct tdma_ioctl_usge) * (u32)arg.usgl_size, GFP_KERNEL);
  if (!arg_usgl) {
    pr_err("tdma: failed to allocate memory for tdma_ioctl_sge");
    return -ENOMEM;
  }  

  if (copy_from_user(arg_usgl, (void __user *)arg.usgl, sizeof(struct tdma_ioctl_usge) * (u32)arg.usgl_size)) {
    pr_err("tdma: failed to copy tdma_ioctl_sge from userspace");
    kfree(arg_usgl);
    return -EFAULT;
  }

  // Locate PCI device
  struct pci_dev* pdev = pci_get_domain_bus_and_slot(arg.domain, arg.bus, PCI_DEVFN(arg.device, arg.function));
  if (!pdev) {
    pr_err("tdma: pci device not found: %04x:%02x:%02x.%x", arg.domain, arg.bus, arg.device, arg.function);
    kfree(arg_usgl);
    return -ENODEV;
  }

  // Validate the BAR
  if (arg.bar >= PCI_STD_NUM_BARS) {
    pr_err("tdma: BAR out of bounds: %s %u", dev_name(&pdev->dev), arg.bar);
    pci_dev_put(pdev);
    kfree(arg_usgl);
    return -EINVAL;
  }

  resource_size_t bar_start = pci_resource_start(pdev, arg.bar);
  resource_size_t bar_size = pci_resource_len(pdev, arg.bar);
  unsigned long bar_flags = pci_resource_flags(pdev, arg.bar);

  if (!bar_size || !(bar_flags & IORESOURCE_MEM)) {
    pr_err("tdma: invalid BAR %u: device=%s size=%llu flags=0x%lx", arg.bar, dev_name(&pdev->dev), bar_size, bar_flags);
    pci_dev_put(pdev);
    kfree(arg_usgl);
    return -EINVAL;
  }

  // Validate user sgl and calculate total dmabuf size
  resource_size_t dmabuf_size = 0;
  for(int i = 0; i < arg.usgl_size; i++) {
    struct tdma_ioctl_usge usge = arg_usgl[i];

    if (usge.offset >= bar_size || usge.offset % PAGE_SIZE != 0 ||
        usge.size == 0 || usge.size > SZ_2G || usge.size % PAGE_SIZE != 0 ||
        (usge.offset + usge.size) > bar_size) {
      pr_err("tdma: invalid user sge");
      pci_dev_put(pdev);
      kfree(arg_usgl);
      return -EINVAL;
    }

    dmabuf_size += usge.size;
  }

  // Create DMABuf
  struct tdma_priv* priv = kzalloc(sizeof(*priv), GFP_KERNEL);
  if (!priv) {
    pr_err("tdma: failed to allocate memory for tdma_priv");
    pci_dev_put(pdev);
    kfree(arg_usgl);
    return -ENOMEM;
  }  

  priv->pdev = pdev;
  priv->bar = arg.bar;
  priv->bar_start = bar_start;
  priv->bar_size = bar_size;
  priv->usgl_size = arg.usgl_size;
  priv->usgl = arg_usgl;

  DEFINE_DMA_BUF_EXPORT_INFO(exp); // sets exp_name to module name and owner to THIS_MODULE (for refcounting, can't rmmod while dmabufs are alive)
  exp.ops = &tdma_ops;
  exp.priv = priv;
  exp.size = dmabuf_size;

  struct dma_buf* dmabuf = dma_buf_export(&exp);
  if (IS_ERR(dmabuf)) {
    pr_err("tdma: failed to export dmabuf: %ld", PTR_ERR(dmabuf));
    kfree(priv);
    pci_dev_put(pdev);
    kfree(arg_usgl);
    return PTR_ERR(dmabuf);
  }
  // DMABuf is created succesfully, the ownership of priv, usgl and pdev is now transfered to it and will be released when dmabuf is released

  int dmabuf_fd = dma_buf_fd(dmabuf, O_CLOEXEC); // TODO: flags should be from userspace. for now cloexec as a sane default
  if (dmabuf_fd < 0) {
    pr_err("tdma: failed to create dmabuf fd: %d", dmabuf_fd);
    dma_buf_put(dmabuf);
    return dmabuf_fd; // if dmabuf_fd < 0 it's the error code instead of an fd
  }
  // DMABuf-fd is created succesfully. It looks like the ownership of dmabuf is now transfered to fd??? Somehow dma_buf_put here is not needed

  pr_debug("tdma: %s exported as dmabuf (fd=%d)", dev_name(&pdev->dev), dmabuf_fd);

  arg.out_fd = dmabuf_fd;

  if (copy_to_user((void __user *)uarg, &arg, sizeof(arg))) {
    pr_err("tdma: Failed to copy ioctl result to userspace");
    // Attempting to deallocate the dmabuf or put the fd here is not done on purpose because it is **NOT** safe to do, as for example another thread
    // might've duplicated the fd and in that case there would be a use after free.
    return -EFAULT;
  }

  return 0;
}

static const struct file_operations tdma_fops = {
  .owner = THIS_MODULE,
  .unlocked_ioctl = tdma_ioctl,
};

static struct miscdevice tdma_misc = {
  .minor = MISC_DYNAMIC_MINOR,
  .name = "tdma",
  .fops = &tdma_fops,
};

static int __init tdma_init(void) {
  int ret = misc_register(&tdma_misc);

  if (ret)
    pr_err("tdma: load error: %i", ret);
  else
    pr_info("tdma: module ready");

  return ret;
}

static void __exit tdma_exit(void) {
  misc_deregister(&tdma_misc);
  pr_info("tdma: unloaded");
}

module_init(tdma_init);
module_exit(tdma_exit);

MODULE_IMPORT_NS("DMA_BUF");

MODULE_LICENSE("Dual MIT/GPL"); // kernel will refuse to load without gpl here...
MODULE_AUTHOR("uuuvn");
MODULE_DESCRIPTION("tdma");
