import torch as t
import torchvision as tv
import torchnet as tnt
from torch.utils import data
from TransformNet import TransforeNet
import utils
from VGG16 import Vgg16
from torch.nn import functional as F
import tqdm
from config import opt


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = utils.Visualizer(opt.env)

    # 数据加载
    transfroms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 255)
    ])
    dataset = tv.datasets.ImageFolder(opt.data_root, transfroms)
    dataloader = data.DataLoader(dataset, opt.batch_size)

    # 转换网络
    transformer = TransforeNet()
    if opt.model_path:
        transformer.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    transformer.to(device)

    # 损失网络 Vgg16
    vgg = Vgg16().eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    # 优化器
    optimizer = t.optim.Adam(transformer.parameters(), opt.lr)

    # 获取风格图片的数据
    style = utils.get_style_data(opt.style_path)
    vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
    style = style.to(device)

    # 风格图片的gram矩阵
    with t.no_grad():
        features_style = vgg(style)
        gram_style = [utils.gram_matrix(y) for y in features_style]

    # 损失统计
    style_meter = tnt.meter.AverageValueMeter()
    content_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.epoches):
        content_meter.reset()
        style_meter.reset()

        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):

            # 训练
            optimizer.zero_grad()
            x = x.to(device)
            y = transformer(x)
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            features_y = vgg(y)
            features_x = vgg(x)

            # content loss
            content_loss = opt.content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)

            # style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gram_y = utils.gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            # 损失平滑
            content_meter.add(content_loss.item())
            style_meter.add(style_loss.item())

            if (ii + 1) % opt.plot_every == 0:

                # 可视化
                vis.plot('content_loss', content_meter.value()[0])
                vis.plot('style_loss', style_meter.value()[0])
                # 因为x和y经过标准化处理(utils.normalize_batch)，所以需要将它们还原
                vis.img('output', (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                vis.img('input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

        # 保存visdom和模型
        vis.save([opt.env])
        t.save(transformer.state_dict(), 'checkpoints/%s_style.pth' % epoch)


@t.no_grad()
def stylize(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    # 图片处理
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()

    # 模型
    style_model = TransforeNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    style_model.to(device)

    # 风格迁移与保存
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)


if __name__ == '__main__':
    import fire

    fire.Fire()
