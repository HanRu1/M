from setuptools import setup, find_packages

setup(
    name='mm',  # 包的名称
    version='0.1.0',  # 包的版本
    author='Your Name',  # 你的名字或者团队名称
    author_email='your.email@example.com',  # 你的邮箱或者团队邮箱
    description='A Python package for medical image metrics and operations',  # 包的简短描述
    long_description=open('README.md').read(),  # 从README.md文件读取更详细的描述
    long_description_content_type='text/markdown',  # 描述内容的格式
    url='https://github.com/HanRu1/M.git',  # 项目的URL，通常是GitHub的URL
    packages=find_packages(),  # 自动查找所有包和子包
    classifiers=[
        'Development Status :: 3 - Alpha',  # 开发状态，从1（计划）到7（停止）
        'Intended Audience :: Developers',  # 目标用户
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # 开源协议
        'Programming Language :: Python :: 3',  # 支持的Python版本
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='medical imaging metrics numpy scipy skimage sklearn',  # 关键词
    python_requires='>=3.7',  # 对Python的最低版本要求
    install_requires=[
        'numpy',
        'scipy',
        'SimpleITK',
        'scikit-image',
        'scikit-learn',
        'sewar',  # 这个包你在代码中使用了，但没有提及
    ],  # 依赖的包
    include_package_data=True,  # 包含在包里的数据文件
    zip_safe=False
)
